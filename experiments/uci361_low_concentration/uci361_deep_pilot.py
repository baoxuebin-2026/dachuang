"""Exploratory temporal MLP against equal-feature baselines, device holdout."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
ARCHIVE = (ROOT / 'uci361_newtrial/source.zip') if (ROOT / 'uci361_newtrial/source.zip').exists() else ROOT / 'source.zip'
SHA = 'e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167'
RX = re.compile(r'data1/B([1-5])_G(CO|Me|Ea|Ey)_F(\d{3})_R([1-4])\.txt')
LABELS = ['CO','Me','Ea','Ey']


def extract(stream):
    valid = {}
    last_second = -1
    for line in stream:
        parts = line.split()
        second = int(float(parts[0]))
        if second >= 90:
            break
        if second == last_second:
            continue
        last_second = second
        if not (25 <= second < 45 or 50 <= second < 90):
            continue
        x = np.fromiter((float(w) for w in parts[1:]), float, count=8)
        if x.shape == (8,) and np.isfinite(x).all() and (x > 0).all():
            valid[second] = np.log(x)
    background = [valid[t] for t in range(25,45) if t in valid]
    if len(background) < 17:
        raise ValueError('Insufficient background observations')
    center = np.median(background, axis=0)
    time_bins = []
    for start in range(50,90,2):
        vals = [valid[t] for t in (start,start+1) if t in valid]
        if not vals:
            raise ValueError(f'Missing time bin {start}')
        time_bins.append(np.median(vals,axis=0)-center)
    curve = np.array(time_bins)
    return center, np.median(curve,axis=0), curve.ravel()


def metrics(y,pred,mask,board):
    yt=y[mask]; yp=pred[mask]; bb=board[mask]
    mat=confusion_matrix(yt,yp,labels=LABELS)
    return {'n':int(mask.sum()),'macro_f1':float(f1_score(yt,yp,labels=LABELS,average='macro',zero_division=0)),
            'co_recall':float(mat[0,0]/sum(mat[0])),'methane_recall':float(mat[1,1]/sum(mat[1])),
            'confusion':mat.tolist(),
            'per_board':{str(k):float(f1_score(yt[bb==k],yp[bb==k],labels=LABELS,average='macro',zero_division=0)) for k in range(1,6)}}


def model(name):
    if name.endswith('logreg'):
        estimator=LogisticRegression(max_iter=2000,C=1)
    elif name.endswith('rbf'):
        estimator=SVC(C=1,gamma='scale')
    else:
        estimator=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='adam',
                  alpha=.1,learning_rate_init=.001,early_stopping=True,validation_fraction=.2,
                  n_iter_no_change=25,max_iter=500,random_state=2026)
    return make_pipeline(StandardScaler(),estimator)


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()!=SHA:
        raise ValueError('Unexpected source archive')
    meta=[]; bases=[]; means=[]; curves=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        names=sorted(n for n in z.namelist() if RX.fullmatch(n))
        assert len(names)==640
        for i,n in enumerate(names,1):
            b,g,lev,repeat=RX.fullmatch(n).groups()
            base,mean,curve=extract(z.open(n))
            meta.append({'file':n,'board':int(b),'gas':g,'level':int(lev)//10,'repeat':int(repeat)})
            bases.append(base);means.append(mean);curves.append(curve)
            if i%160==0:print('loaded',i,flush=True)
    board=np.array([m['board'] for m in meta]);y=np.array([m['gas'] for m in meta]);level=np.array([m['level'] for m in meta]);
    x={'baseline_logreg':np.stack(bases),'delta_logreg':np.stack(means),
       'temporal_logreg':np.stack(curves),'temporal_rbf':np.stack(curves),'temporal_mlp':np.stack(curves)}
    results={}; predictions=[]
    for name,xx in x.items():
        for task in ('low_held_board','all_held_board'):
            pred=np.full(len(y),'',dtype='<U2')
            mask=np.zeros(len(y),bool)
            fit_steps={}
            for held in range(1,6):
                training=(board!=held) & ((level>=4) if task=='low_held_board' else True)
                testing=(board==held) & ((level<=3) if task=='low_held_board' else True)
                pipeline=model(name)
                if name=='temporal_mlp':
                    numeric=np.array([LABELS.index(label) for label in y])
                    pipeline.fit(xx[training],numeric[training])
                    pred[testing]=np.array(LABELS)[pipeline.predict(xx[testing])]
                else:
                    pipeline.fit(xx[training],y[training]);pred[testing]=pipeline.predict(xx[testing])
                mask|=testing
                if name=='temporal_mlp':
                    nn=pipeline[-1];fit_steps[str(held)]={'epochs':int(nn.n_iter_),'internal_best_validation_score':float(nn.best_validation_score_)}
            results[f'{name}_{task}']={**metrics(y,pred,mask,board),'fitting':fit_steps}
            for idx in np.flatnonzero(mask):
                predictions.append({**meta[idx],'task':task,'model':name,'prediction':pred[idx]})
            s=results[f'{name}_{task}'];print(name,task,'F1',round(s['macro_f1'],3),'CO',round(s['co_recall'],3),'Me',round(s['methane_recall'],3),'worst',round(min(s['per_board'].values()),3),flush=True)
    output=ROOT/'uci361_temporal_mlp_exploratory_2026-10-19.json'
    output.write_text(json.dumps({'protocol':'Exploratory after previous UCI 361 test inspection. Full trials, time bins 2 s over 50-90 s, leave full device out; low task trains concentration ranks 4-10, tests 1-3. Architecture and seed fixed in script before this run.','source_sha256':SHA,'trials':len(meta),'by_board':dict(Counter(board.tolist())),'results':results},ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'uci361_temporal_mlp_predictions_2026-10-19.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]),lineterminator='\n');w.writeheader();w.writerows(predictions)
    print('saved',output,flush=True)


if __name__=='__main__':main()
