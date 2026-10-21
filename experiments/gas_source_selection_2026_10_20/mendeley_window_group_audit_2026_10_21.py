"""Frozen source-file unit audit on 186 CH4/CO/H2S controlled mixtures.

Mendeley 10.17632/2mpx85xt6g.1 supplies one recording per concentration
recipe and three response channels; each recording contributes four disjoint
100-sample gas windows. Train ExtraTrees(300,leaf1,.8) on response/delta,
without explicit baseline feature. Compare random 5-fold windows against
StratifiedGroupKFold 5-fold holding out entire physical files. Three seeds
(11,21,37) fixed. Target is filename's middle concentration (CO, 0..2500).
Physical-file MAE uses mean of four out-of-fold window predictions. Success:
group MAE >2 times random in every seed. No inference to field CO sensing or
claim that data authors used leakage. Independent recipe count is 186.

Usage: python mendeley_window_group_audit_2026_10_21.py /path/to/extracted
RAR sha256 bc5acaf07f7560a78aa302e2c37627e073b25fa45c3175c70fd67e8ce10efc19.
"""
from pathlib import Path
import json,sys
import numpy as np
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesRegressor

files=sorted(Path(sys.argv[1]).rglob('*.txt'))
assert len(files)==186,len(files)
features=[];labels=[];units=[]
for fid,p in enumerate(files):
    ch4,co,h2s=map(int,p.stem.split('-'))
    with p.open() as f:
        signal=np.array([[float(v) for v in f.readline().split()] for _ in range(500)])
    assert signal.shape==(500,3) and np.isfinite(signal).all()
    baseline=np.median(signal[:80],axis=0)
    for a,b in ((100,200),(200,300),(300,400),(400,500)):
        current=np.median(signal[a:b],axis=0)
        features.append(np.r_[np.log1p(current),np.sign(current-baseline)*np.log1p(abs(current-baseline)),(current-baseline)/(abs(baseline)+1000)])
        labels.append(co);units.append(fid)
X=np.stack(features);y=np.asarray(labels);units=np.asarray(units)
out={'protocol':__doc__,'n_files':186,'n_windows':len(y),'results':{}}
for seed in (11,21,37):
    row={'seed':seed}
    for grouped in (False,True):
        cv=(StratifiedGroupKFold(5,shuffle=True,random_state=seed) if grouped else StratifiedKFold(5,shuffle=True,random_state=seed))
        pred=np.full(len(y),np.nan)
        for train,test in cv.split(X,y,units if grouped else None):
            model=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=1,max_features=.8,random_state=21,n_jobs=1)
            pred[test]=np.clip(model.fit(X[train],y[train]).predict(X[test]),0,2500)
        assert np.isfinite(pred).all()
        mae=np.mean([abs(y[units==u][0]-pred[units==u].mean()) for u in range(186)])
        row['file_group_mae' if grouped else 'random_window_mae']=round(float(mae),2)
    row['ratio']=round(row['file_group_mae']/row['random_window_mae'],3)
    print(row,flush=True);out['results'][str(seed)]=row
out['pass']=all(v['ratio']>2 for v in out['results'].values())
print('PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
