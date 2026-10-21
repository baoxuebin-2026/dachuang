"""Frozen file-unit test of response/recovery kinetics in CO/CH4/H2S.

Each of 186 physical mixture recordings is one independent unit. First 50
samples are baseline; gas is supplied at 50..250, recovery at 250..500.
Baseline and three exposure medians form the existing 12D static reference.
Candidate: baseline and ten signed-log response/recovery bin medians (33D),
also response minus recovery summaries for each sensor (36D). Both use the
same ExtraTrees(300, max_features=.8), and SVR(C=100) as sensitivity.
Three prechosen shuffled stratified fivefold complete-file splits (11,21,37).
Primary CO MAE; high CO >=2000 ppm positive underestimation is secondary.
Pass only if richer kinetics improves MAE >=10% and high underestimation is
not worse in EVERY split seed versus static ExtraTrees. This is an
exploratory method test, not independent confirmation or novelty evidence.
"""
from pathlib import Path
import json
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

paths = sorted(Path(sys.argv[1]).rglob('*.txt'))
assert len(paths) == 186
xs, ys = [], []
bins = ((50,75),(75,100),(100,150),(150,200),(200,250),
        (250,275),(275,300),(300,350),(350,400),(400,500))
for p in paths:
    with p.open() as f:
        a = np.array([[float(z) for z in f.readline().split()] for _ in range(500)])
    base = np.median(a[:50],axis=0)
    sglog = lambda x: np.sign(x)*np.log1p(abs(x))
    short = np.r_[sglog(base), *(sglog(np.median(a[t:t+50],axis=0)-base)
                                  for t in (100,200,300))]
    steps = np.stack([sglog(np.median(a[lo:hi],axis=0)-base) for lo,hi in bins])
    long = np.r_[sglog(base),steps.ravel(),
                 steps[4]-steps[7]]
    xs.append((short,long));ys.append(int(p.stem.split('-')[1]))
X1=np.array([x[0] for x in xs]);X2=np.array([x[1] for x in xs]);y=np.array(ys)
result={'protocol':__doc__,'n_files':len(y),'results':{}}
for seed in (11,21,37):
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    preds={k:np.full(len(y),np.nan) for k in ('static_et','kinetics_et','static_svr','kinetics_svr')}
    for tr,te in cv.split(X1,y):
        for label,X in [('static',X1),('kinetics',X2)]:
            z=StandardScaler().fit(X[tr]); a,b=z.transform(X[tr]),z.transform(X[te])
            preds[label+'_et'][te]=ExtraTreesRegressor(n_estimators=300,
                max_features=.8,random_state=21,n_jobs=1).fit(a,y[tr]).predict(b)
            preds[label+'_svr'][te]=1000*SVR(C=100,epsilon=.1,gamma='scale').fit(
                a,y[tr]/1000).predict(b)
    stats={}
    for label,p in preds.items():
        p=np.clip(p,0,2500)
        stats[label]={'mae':round(float(np.mean(abs(p-y))),2),
                      'high_under':round(float(np.mean(np.maximum(0,y[y>=2000]-p[y>=2000]))),2)}
    stats['pass']=bool(stats['kinetics_et']['mae']<=.9*stats['static_et']['mae']
                       and stats['kinetics_et']['high_under']<=stats['static_et']['high_under'])
    result['results'][str(seed)]=stats
    print(seed,stats,flush=True)
result['pass']=all(d['pass'] for d in result['results'].values())
print('GLOBAL PASS',result['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
