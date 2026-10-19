"""Post hoc initialisation and augmentation strength sensitivity."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score
import uci361_tinycnn_pilot as p

ROOT=Path(__file__).resolve().parent
SETTINGS=[('four_factors_seed2025',(.25,.5,.75,1.),2025),
          ('four_factors_seed2027',(.25,.5,.75,1.),2027),
          ('three_factors_seed2026',(.5,.75,1.),2026),
          ('two_factors_seed2026',(.5,1.),2026)]

def main():
    x,y,b,levels,paths=p.get_trials()
    res={}
    for tag,factors,seed in SETTINGS:
        pred=np.full(len(y),-1)
        for held in range(1,6):
            train=(b!=held)&(levels>=4)
            test=(b==held)&(levels<=3)
            aug=np.concatenate([x[train]*factor for factor in factors])
            yy=np.tile(y[train],len(factors))
            pred[test],_=p.train_predict(aug,yy,x[test],seed=seed)
            print(tag,'held',held,'correct',int((pred[test]==y[test]).sum()),'of',int(test.sum()),flush=True)
        sel=pred>=0
        cm=confusion_matrix(y[sel],pred[sel],labels=range(4))
        res[tag]={'factors':factors,'seed':seed,'macro_f1':float(f1_score(y[sel],pred[sel],average='macro',labels=range(4))),
                  'co_recall':float(cm[0,0]/cm[0].sum()),'methane_recall':float(cm[1,1]/cm[1].sum()),
                  'worst_device_f1':float(min(f1_score(y[sel&(b==k)],pred[sel&(b==k)],average='macro',labels=range(4),zero_division=0) for k in range(1,6))),
                  'confusion':cm.tolist()}
    dest=ROOT/'uci361_cnn_posthoc_robustness_2026-10-19.json'
    dest.write_text(json.dumps({'status':'Post hoc sensitivity on already inspected five devices. No independent holdout remains.','settings':res},ensure_ascii=False,indent=2)+'\n')
    print('saved',dest,flush=True)
    print({k:(v['macro_f1'],v['co_recall'],v['methane_recall'],v['worst_device_f1']) for k,v in res.items()},flush=True)

if __name__=='__main__':main()
