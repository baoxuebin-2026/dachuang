"""Compact trainable temporal CNN pilot with leave-one-board-out testing."""
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parent
ARCHIVE=(ROOT/'uci361_newtrial/source.zip') if (ROOT/'uci361_newtrial/source.zip').exists() else ROOT/'source.zip'
NAME=re.compile(r'data1/B([1-5])_G(CO|Me|Ea|Ey)_F(\d{3})_R([1-4])\.txt')
LABELS=['CO','Me','Ea','Ey']
K=16
WIN=3
BLOCK=20-WIN+1
SHAPES=((WIN*8,K),(K,),(K*2,4),(4,))
SIZES=[int(np.prod(shape)) for shape in SHAPES]


def get_trials():
    expected='e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167'
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()!=expected:
        raise ValueError('Unexpected source archive')
    y=[];boards=[];ranks=[];paths=[];xs=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        names=sorted(n for n in z.namelist() if NAME.fullmatch(n))
        assert len(names)==640
        for i,name in enumerate(names,1):
            board,gas,level,_=NAME.fullmatch(name).groups()
            seconds={};last=-1
            with z.open(name) as stream:
                for line in stream:
                    fields=line.split(); sec=int(float(fields[0]))
                    if sec>=90:break
                    if sec==last:continue
                    last=sec
                    if not (25<=sec<45 or 50<=sec<90):continue
                    val=np.fromiter((float(w) for w in fields[1:]),float,count=8)
                    if val.shape==(8,) and np.isfinite(val).all() and (val>0).all(): seconds[sec]=np.log(val)
            baseline=np.median([seconds[t] for t in range(25,45) if t in seconds],axis=0)
            curve=[]
            for t in range(50,90,2):
                good=[seconds[s] for s in (t,t+1) if s in seconds]
                if not good:raise ValueError(name)
                curve.append(np.median(good,axis=0)-baseline)
            xs.append(curve);y.append(LABELS.index(gas));boards.append(int(board));ranks.append(int(level)//10);paths.append(name)
            if i%160==0:print('read',i,flush=True)
    return np.array(xs),np.array(y),np.array(boards),np.array(ranks),paths


def unpack(p):
    indices=np.cumsum([0,*SIZES]);return [p[indices[i]:indices[i+1]].reshape(s) for i,s in enumerate(SHAPES)]


def windows(x):
    return np.lib.stride_tricks.sliding_window_view(x,WIN,axis=1).transpose(0,1,3,2).reshape(len(x),BLOCK,WIN*8)


def objective(p, x, y, l2=.02):
    w,b,v,c=unpack(p)
    z=x@w+b
    h=np.maximum(z,0.)
    pooling=np.concatenate((h[:,:9].mean(axis=1),h[:,9:].mean(axis=1)),axis=1)
    logits=pooling@v+c
    logits-=logits.max(axis=1,keepdims=True)
    probs=np.exp(logits);probs/=probs.sum(axis=1,keepdims=True)
    loss=-np.log(np.maximum(probs[np.arange(len(y)),y],1e-15)).mean()+l2*(np.square(w).sum()+np.square(v).sum())/2
    dz=probs;dz[np.arange(len(y)),y]-=1;dz/=len(y)
    gv=pooling.T@dz+l2*v;gc=dz.sum(axis=0)
    dh_pool=dz@v.T
    dh=np.zeros_like(h)
    dh[:,:9]=dh_pool[:,:K,None].transpose(0,2,1)/9
    dh[:,9:]=dh_pool[:,K:,None].transpose(0,2,1)/9
    dpre=dh*(z>0)
    gw=np.einsum('ntd,ntk->dk',x,dpre)+l2*w
    gb=dpre.sum(axis=(0,1))
    return loss,np.concatenate([q.ravel() for q in (gw,gb,gv,gc)])


def gradient_check():
    rng=np.random.default_rng(42)
    xx=rng.normal(size=(6,BLOCK,WIN*8));yy=np.array([0,1,2,3,0,1])
    p=rng.normal(scale=.08,size=sum(SIZES))
    loss,grad=objective(p,xx,yy)
    for idx in (0,12,400,len(p)-1):
        trial=p.copy();trial[idx]+=1e-5
        delta=(objective(trial,xx,yy)[0]-loss)/1e-5
        if abs(delta-grad[idx])>1e-4:raise AssertionError((idx,delta,grad[idx]))


def train_predict(train_x,train_y,test_x,seed=2026):
    # Channel-wise standardization is fitted only on training devices.
    mu=train_x.mean(axis=(0,1),keepdims=True)
    sd=np.maximum(train_x.std(axis=(0,1),keepdims=True),.03)
    a=windows(np.clip((train_x-mu)/sd,-8,8))
    b=windows(np.clip((test_x-mu)/sd,-8,8))
    rng=np.random.default_rng(seed)
    w=rng.normal(0,np.sqrt(2/(8*WIN)),SHAPES[0]); bias=np.full(SHAPES[1],.01)
    v=rng.normal(0,np.sqrt(2/(K*2)),SHAPES[2]); c=np.zeros(SHAPES[3])
    init=np.concatenate([x.ravel() for x in (w,bias,v,c)])
    res=minimize(objective,init,args=(a,train_y),jac=True,method='L-BFGS-B',
                 options={'maxiter':160,'ftol':1e-7})
    w,bias,v,c=unpack(res.x)
    h=np.maximum(b@w+bias,0)
    pooling=np.concatenate((h[:,:9].mean(axis=1),h[:,9:].mean(axis=1)),axis=1)
    return (pooling@v+c).argmax(axis=1),{'iterations':res.nit,'converged':res.success,'train_loss':float(res.fun)}


def main():
    gradient_check()
    sha=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert sha=='e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167'
    x,y,boards,levels,paths=get_trials()
    output={};details=[]
    for task in ('low_held_board','all_held_board'):
        predicted=np.full(len(y),-1)
        fitting={}
        for held in range(1,6):
            train=(boards!=held)&((levels>=4) if task=='low_held_board' else True)
            test=(boards==held)&((levels<=3) if task=='low_held_board' else True)
            pred,stats=train_predict(x[train],y[train],x[test])
            predicted[test]=pred;fitting[str(held)]=stats
            print(task,'device',held,stats,flush=True)
        subset=predicted!=-1
        mat=confusion_matrix(y[subset],predicted[subset],labels=range(4))
        per={str(k):float(f1_score(y[subset&(boards==k)],predicted[subset&(boards==k)],labels=range(4),average='macro',zero_division=0)) for k in range(1,6)}
        output[task]={'n':int(subset.sum()),'macro_f1':float(f1_score(y[subset],predicted[subset],labels=range(4),average='macro',zero_division=0)),
                      'co_recall':float(mat[0,0]/mat[0].sum()),'methane_recall':float(mat[1,1]/mat[1].sum()),
                      'confusion':mat.tolist(),'per_board':per,'fitting':fitting}
        details.extend({'file':paths[i],'board':int(boards[i]),'level':int(levels[i]),'truth':LABELS[y[i]],'prediction':LABELS[predicted[i]],'task':task} for i in np.where(subset)[0])
    dest=ROOT/'uci361_tinycnn_exploratory_2026-10-19.json'
    dest.write_text(json.dumps({'method':'Trainable 1D conv: 16 kernels length 3, two-half mean pooling, softmax; fixed weight decay 0.02, max 160 L-BFGS steps; train-only scaling. Exploration on previously inspected UCI361 dataset.','source_sha256':sha,'result':output},ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'uci361_tinycnn_predictions_2026-10-19.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]),lineterminator='\n');writer.writeheader();writer.writerows(details)
    print('saved',dest,flush=True)
    print({k:{'F1':v['macro_f1'],'CO':v['co_recall'],'Me':v['methane_recall'],'worst_device':min(v['per_board'].values())} for k,v in output.items()},flush=True)

if __name__=='__main__':main()
