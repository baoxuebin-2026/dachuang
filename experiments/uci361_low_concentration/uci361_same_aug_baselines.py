"""Fixed classical baselines with identical three-factor augmentation and splits."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import uci361_tinycnn_pilot as data

ROOT=Path(__file__).resolve().parent
FACTORS=(.5,.75,1.)
LABELS=data.LABELS


def summary(y, pred, b, mask):
    mat=confusion_matrix(y[mask],pred[mask],labels=range(4))
    return {'n':int(mask.sum()),'macro_f1':float(f1_score(y[mask],pred[mask],labels=range(4),average='macro',zero_division=0)),
       'co_recall':float(mat[0,0]/mat[0].sum()),'methane_recall':float(mat[1,1]/mat[1].sum()),
       'co_false_positive_count':int(mat[1:,0].sum()),'confusion':mat.tolist(),
       'per_board_f1':{str(k):float(f1_score(y[mask&(b==k)],pred[mask&(b==k)],labels=range(4),average='macro',zero_division=0)) for k in range(1,6)}}


def main():
    x,y,b,levels,paths=data.get_trials()
    models={
        'temporal_logistic':LogisticRegression(max_iter=3000,C=1),
        'temporal_rbf_svm':SVC(C=1,gamma='scale'),
    }
    preds=[];results={}
    for name,estimator in models.items():
        pred=np.full(len(y),-1)
        for held in range(1,6):
            train=(b!=held)&(levels>=4)
            test=(b==held)&(levels<=3)
            augmented=np.concatenate([x[train]*factor for factor in FACTORS])
            target=np.tile(y[train],len(FACTORS))
            classifier=make_pipeline(StandardScaler(),estimator)
            classifier.fit(augmented.reshape(len(augmented),-1),target)
            pred[test]=classifier.predict(x[test].reshape(test.sum(),-1))
            print(name,'board',held,'correct',int(sum(y[test]==pred[test])),'/',int(test.sum()),flush=True)
        mask=pred!=-1
        results[name]=summary(y,pred,b,mask)
        preds.extend({'file':paths[i],'board':int(b[i]),'level':int(levels[i]),'truth':LABELS[y[i]],
              'prediction':LABELS[pred[i]],'method':name} for i in np.flatnonzero(mask))
    out=ROOT/'uci361_three_scale_classical_baselines_2026-10-19.json'
    out.write_text(json.dumps({'protocol':'Exploratory fixed-parameter, identical three-scale augmentation and per-device low-concentration holdout; test data previously inspected.',
         'factors':FACTORS,'results':results},ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'uci361_three_scale_classical_predictions_2026-10-19.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(preds[0]),lineterminator='\n');w.writeheader();w.writerows(preds)
    print('saved',out,flush=True)
    print({k:(v['macro_f1'],v['co_recall'],v['methane_recall'],v['co_false_positive_count'],min(v['per_board_f1'].values())) for k,v in results.items()},flush=True)

if __name__=='__main__':main()
