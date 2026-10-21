"""Frozen pilot: sign-conditioned CO mixture regression, one file per sample.

Matched CO=0 vs 2500 ppm at identical CH4/H2S shows that H2S=0 has
negative MOX-channel CO response while many H2S>0 cells reverse direction.
Train a probabilistic H2S-presence classifier, two ExtraTrees CO experts on
the H2S=0 and H2S>0 subsets, then blend their CO predictions with predicted
probability. No test H2S label is used. Baselines: one global ExtraTrees(300)
CO model on the same 12 response features, and an oracle known-H2S switch
reported only as an upper bound (not a deployable competitor). Three fixed
complete-file stratified 5-fold splits (11,21,37); score per-file CO MAE,
high CO>=2000 ppm underestimation, and H2S=0 subset MAE. Pass only if
non-oracle mixture improves overall MAE >=10% and high underestimation does
not worsen in every seed, with no H2S=0 MAE degradation >10%. This is
exploratory reuse of 186 recipes, not blinded validation or originality.
"""
from pathlib import Path
import json
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

files = sorted(Path(sys.argv[1]).rglob('*.txt'))
assert len(files)==186
xx=[];yy=[];hs=[]
for p in files:
    with p.open() as f:
        a=np.array([[float(t) for t in f.readline().split()] for _ in range(500)])
    baseline=np.median(a[:50],axis=0)
    lg=lambda x:np.sign(x)*np.log1p(abs(x))
    xx.append(np.r_[lg(baseline),*[lg(np.median(a[s:s+50],axis=0)-baseline)
                                   for s in (100,200,300)]])
    _,co,h2s=map(int,p.stem.split('-'));yy.append(co);hs.append(h2s>0)
X=np.array(xx);Y=np.array(yy);H=np.array(hs)
res={'protocol':__doc__,'n_files':len(Y),'results':{}}
for seed in (11,21,37):
    pred={k:np.full(len(Y),np.nan) for k in ('global','sign_conditioned','oracle_switch')}
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,Y):
        reg=lambda:ExtraTreesRegressor(n_estimators=300,max_features=.8,
                                      random_state=21,n_jobs=1)
        clf=ExtraTreesClassifier(n_estimators=300,max_features=.8,
                                 random_state=21,n_jobs=1).fit(X[tr],H[tr])
        global_model=reg().fit(X[tr],Y[tr])
        separate=[reg().fit(X[tr][H[tr]==v],Y[tr][H[tr]==v]) for v in (False,True)]
        pg=global_model.predict(X[te]);p0,p1=[m.predict(X[te]) for m in separate]
        prob=clf.predict_proba(X[te])[:,1]
        pred['global'][te]=pg
        pred['sign_conditioned'][te]=(1-prob)*p0+prob*p1
        pred['oracle_switch'][te]=np.where(H[te],p1,p0)
    stats={}
    for k,p in pred.items():
        p=np.clip(p,0,2500)
        stats[k]={'mae':round(float(np.mean(abs(Y-p))),2),
                  'high_under':round(float(np.mean(np.maximum(0,Y[Y>=2000]-p[Y>=2000]))),2),
                  'h2s_zero_mae':round(float(np.mean(abs(Y[~H]-p[~H]))),2)}
    base=stats['global'];new=stats['sign_conditioned']
    stats['pass']=bool(new['mae']<=.9*base['mae'] and new['high_under']<=base['high_under']
                      and new['h2s_zero_mae']<=1.1*base['h2s_zero_mae'])
    res['results'][str(seed)]=stats
    print(seed,stats,flush=True)
res['pass']=all(row['pass'] for row in res['results'].values())
print('GLOBAL PASS',res['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(res,indent=2))
