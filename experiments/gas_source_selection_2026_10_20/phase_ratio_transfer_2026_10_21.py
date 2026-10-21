"""Frozen CO/humidity heater-phase difference transfer across 14 sensors.

UCI487 day segments 1300 shared exposures per physical unit. For each unit
hold it out, train on six other same-manufacturer units. Compare raw features
(log high phase, log low phase, humidity) against a within-cycle difference
(log high - log low, humidity) to remove multiplicative sensor offset.
Ridge(10), ExtraTrees(150,leaf2) fixed; no target labels in training.
Strong baseline: raw Ridge, raw ET and their 50/50 prediction average.
Predeclared useful criterion: best precommitted ratio model (Ridge, chosen
before results as linear, no model search) beats raw 50/50 on pooled MAE by
>=10%, does not worsen either sensor family or worst target MAE, and does
not increase highCO>=15 ppm positive underestimation. Other ratio models are
diagnostic only. Shared gas schedule across hardware, no VOC validation.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

root=Path(__file__).resolve().parents[2]
d=pd.read_csv(root/'results/uci487_v3/segment_features.csv')
y=np.tile(d.co_reference_ppm.to_numpy(),14);unit=np.repeat(np.arange(1,15),len(d))
raw=np.concatenate([d[[f'log_r{u}_high',f'log_r{u}_low','humidity_pct']].to_numpy() for u in range(1,15)])
ratio=np.c_[raw[:,0]-raw[:,1],raw[:,2]]
models={'ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=10)),
 'extra_trees':lambda:ExtraTreesRegressor(n_estimators=150,min_samples_leaf=2,max_features=1.,random_state=21,n_jobs=1)}
out={'protocol':__doc__,'devices':{}}
for target in range(1,15):
 family=np.arange(1,8) if target<=7 else np.arange(8,15)
 tr=np.where(np.isin(unit,family)&(unit!=target))[0];te=np.where(unit==target)[0]
 predictions={}
 for rep,X in [('raw',raw),('ratio',ratio)]:
  for name,make in models.items():
   predictions[f'{rep}_{name}']=np.clip(make().fit(X[tr],y[tr]).predict(X[te]),0,20)
 predictions['raw_half']=(predictions['raw_ridge']+predictions['raw_extra_trees'])/2
 error={k:round(float(abs(y[te]-p).mean()),3) for k,p in predictions.items()}
 under={k:round(float(np.maximum(0,y[te][y[te]>=15]-p[y[te]>=15]).mean()),3) for k,p in predictions.items()}
 out['devices'][str(target)]={'mae':error,'high_under':under}
 print(target,error,flush=True)
names=('raw_ridge','raw_extra_trees','raw_half','ratio_ridge','ratio_extra_trees')
summary={}
for name in names:
 scores=np.array([out['devices'][str(u)]['mae'][name] for u in range(1,15)])
 high=np.array([out['devices'][str(u)]['high_under'][name] for u in range(1,15)])
 summary[name]={'pooled_mae':round(float(scores.mean()),3),'figaro_mae':round(float(scores[:7].mean()),3),
   'fis_mae':round(float(scores[7:].mean()),3),'worst_mae':round(float(scores.max()),3),
   'high_under':round(float(high.mean()),3)}
out['summary']=summary
a=summary['ratio_ridge'];b=summary['raw_half']
out['pass']=a['pooled_mae']<=.9*b['pooled_mae'] and a['figaro_mae']<=b['figaro_mae'] and a['fis_mae']<=b['fis_mae'] and a['worst_mae']<=b['worst_mae'] and a['high_under']<=b['high_under']
print('SUMMARY',summary,'PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
