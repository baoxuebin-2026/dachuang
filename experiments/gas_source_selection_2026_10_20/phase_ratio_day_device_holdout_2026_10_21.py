"""Frozen after UCI487 device-only phase-ratio signal was observed.

Stronger UCI487 transfer: first 10 days on six same-type source devices
only; evaluate last 3 days on the seventh (new device AND later dates).
Repeat for seven FIGARO and seven FIS targets. No target labels or test-day
data used by training. Baselines raw log phases+humidity Ridge(10), ET(150,
leaf2), and simple 50/50. Candidate within-cycle difference (log high minus
log low)+humidity with same ET. Each held-out device has 300 exposures; same
gas schedule across devices and days, so independent test units are 3 days
and 14 hardware units, not 4200 independent recipes. Success requires ratio
ET pooled MAE >=10% below best fixed raw model, at least 10% below best fixed
raw in EACH manufacturer family, worst target MAE no worse, and pooled highCO
>=15ppm positive underestimation no worse. This stress test is independent in
time from training but representation choice was motivated by UCI487 itself.
"""
from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

root=Path(__file__).resolve().parents[2];d=pd.read_csv(root/'results/uci487_v3/segment_features.csv')
days=sorted(d.day.unique());assert len(days)==13 and len(d)==1300
source_days=np.isin(d.day,days[:10]);target_days=np.isin(d.day,days[10:])
raw=np.concatenate([d[[f'log_r{u}_high',f'log_r{u}_low','humidity_pct']].to_numpy() for u in range(1,15)])
diff=np.c_[raw[:,0]-raw[:,1],raw[:,2]];y=np.tile(d.co_reference_ppm.to_numpy(),14)
unit=np.repeat(np.arange(1,15),len(d));train_time=np.tile(source_days,14);test_time=np.tile(target_days,14)
makers={'ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=10)),
 'extra_trees':lambda:ExtraTreesRegressor(n_estimators=150,min_samples_leaf=2,max_features=1.,random_state=21,n_jobs=1)}
out={'protocol':__doc__,'days_train':days[:10],'days_test':days[10:],'targets':{}}
for target in range(1,15):
 family=np.arange(1,8) if target<=7 else np.arange(8,15)
 tr=np.where(np.isin(unit,family)&(unit!=target)&train_time)[0]
 te=np.where((unit==target)&test_time)[0]
 assert len(tr)==6000 and len(te)==300
 p={}
 for name,make in makers.items():p['raw_'+name]=np.clip(make().fit(raw[tr],y[tr]).predict(raw[te]),0,20)
 p['raw_half']=.5*(p['raw_ridge']+p['raw_extra_trees'])
 p['ratio_extra_trees']=np.clip(makers['extra_trees']().fit(diff[tr],y[tr]).predict(diff[te]),0,20)
 out['targets'][str(target)]={'mae':{k:round(float(abs(y[te]-v).mean()),3) for k,v in p.items()},
   'high_under':{k:round(float(np.maximum(0,y[te][y[te]>=15]-v[y[te]>=15]).mean()),3) for k,v in p.items()}}
 print(target,out['targets'][str(target)],flush=True)
methods=('raw_ridge','raw_extra_trees','raw_half','ratio_extra_trees')
summary={}
for k in methods:
 z=np.array([out['targets'][str(t)]['mae'][k] for t in range(1,15)])
 u=np.array([out['targets'][str(t)]['high_under'][k] for t in range(1,15)])
 summary[k]={'all':round(float(z.mean()),3),'figaro':round(float(z[:7].mean()),3),
   'fis':round(float(z[7:].mean()),3),'worst':round(float(z.max()),3),'high_under':round(float(u.mean()),3)}
out['summary']=summary
baseline={scope:min(summary[k][scope] for k in methods[:-1]) for scope in ('all','figaro','fis','worst','high_under')}
r=summary['ratio_extra_trees']
out['pass']=r['all']<=.9*baseline['all'] and r['figaro']<=.9*baseline['figaro'] and r['fis']<=.9*baseline['fis'] and r['worst']<=baseline['worst'] and r['high_under']<=baseline['high_under']
print('SUMMARY',summary,'BEST FIXED',baseline,'PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
