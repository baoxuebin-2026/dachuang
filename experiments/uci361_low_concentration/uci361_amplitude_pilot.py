"""Physics-motivated response-amplitude augmentation for cross-device low levels."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import uci361_tinycnn_pilot as conv

ROOT=Path(__file__).resolve().parent
FACTORS=(.25,.5,.75,1.0)

def main():
    x,y,boards,levels,paths=conv.get_trials()
    output={};per_file=[]
    for name in ('augmented_logreg','augmented_cnn'):
        preds=np.full(len(y),-1);training={}
        for held in range(1,6):
            tr=(boards!=held)&(levels>=4)
            te=(boards==held)&(levels<=3)
            augmented=np.concatenate([x[tr]*factor for factor in FACTORS])
            targets=np.tile(y[tr],len(FACTORS))
            if name=='augmented_cnn':
                predicted,stats=conv.train_predict(augmented,targets,x[te])
            else:
                pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1))
                pipe.fit(augmented.reshape(len(augmented),-1),targets)
                predicted=pipe.predict(x[te].reshape(te.sum(),-1))
                stats={'training_examples':len(augmented)}
            preds[te]=predicted;training[str(held)]=stats
            print(name,'held',held,'accuracy',float(np.mean(y[te]==predicted)),flush=True)
        sel=preds!=-1; mat=confusion_matrix(y[sel],preds[sel],labels=range(4))
        output[name]={'n':int(sel.sum()),'macro_f1':float(f1_score(y[sel],preds[sel],average='macro',labels=range(4))),
             'co_recall':float(mat[0,0]/mat[0].sum()),'methane_recall':float(mat[1,1]/mat[1].sum()),
             'per_board':{str(i):float(f1_score(y[sel&(boards==i)],preds[sel&(boards==i)],average='macro',labels=range(4),zero_division=0)) for i in range(1,6)},
             'confusion':mat.tolist(),'training':training}
        per_file.extend({'file':paths[i],'board':int(boards[i]),'truth':conv.LABELS[y[i]],
             'prediction':conv.LABELS[preds[i]],'model':name} for i in np.where(sel)[0])
    dest=ROOT/'uci361_amplitude_augmented_exploratory_2026-10-19.json'
    dest.write_text(json.dumps({'status':'Exploratory after seeing unaugmented model performance; 1D log-response amplitude factors 0.25,0.5,0.75,1 for train devices only; training set ranks >=4; test on held device ranks <=3.','results':output},ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'uci361_amplitude_augmented_predictions_2026-10-19.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_file[0]),lineterminator='\n');w.writeheader();w.writerows(per_file)
    print('saved',dest,flush=True)
    print({k:{'f1':v['macro_f1'],'co':v['co_recall'],'me':v['methane_recall'],'worst':min(v['per_board'].values())} for k,v in output.items()},flush=True)

if __name__=='__main__':main()
