"""Frozen paired-response alignment followed by 20-CO-residual fit.

Data, source Ridge(10), training/test and 20 stratified pilot recipes follow
paired_response_map_pilot_2026_10_21.py. For each source/target physically
matched recipe, source-target signed-log response difference is computed.
Featurewise median of 20 paired differences aligns target features to the
source sensor. Then fit Ridge(10) residuals from 20 target CO labels to the
aligned source model predictions. Holdout test only once; three fixed seeds
(21,37,59) describe pilot sampling variation, not independent test data.
Compare raw-source residual Ridge20, map-only, direct target Ridge20. Primary
success: map-then-residual mean MAE <=90% of BEST raw-residual/map-only
baseline on each dataset, no individual target >10% worse than its own best
baseline, and mean CO>=1500ppb positive underestimation not worse. 6821340
and 15423482 datasets already seen in ideation, so even success is not blind.
"""
from pathlib import Path
import numpy as np,json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

R=Path(__file__).parent;out={'protocol':__doc__,'datasets':{}}
log=lambda v:np.sign(v)*np.log1p(abs(v))
for name,filename,source,targets in [
 ('6821340','exposures_6821340.npz','A',['B','C']),
 ('15423482','exposures_15423482_first.npz','54118801',['54120037','54120457','54156953','84746449','84748697','84749569'])]:
 data=np.load(R/filename)
 def get(uid):
  if name=='6821340':return data[uid+'_train_x'],data[uid+'_train_y'],data[uid+'_test_x'],data[uid+'_test_y']
  return data[uid+'_x'],data[uid+'_y'],None,None
 sx,sy,_,_=get(source)
 tr=np.arange(500) if name=='6821340' else np.where(sy[:,0]<=100)[0]
 pool=np.arange(500,700) if name=='6821340' else np.where((sy[:,0]>100)&(sy[:,0]<160))[0]
 scaler=StandardScaler().fit(log(sx[tr]));model=Ridge(alpha=10).fit(scaler.transform(log(sx[tr])),sy[tr,1])
 devices={}
 for uid in targets:
  tx,ty,tx_test,ty_test=get(uid)
  x_test=tx_test if name=='6821340' else tx[ty[:,0]>=160]
  y_test=ty_test[:,1] if name=='6821340' else ty[ty[:,0]>=160,1]
  voc=ty[pool,4 if name=='6821340' else 2];a=pool[voc<=np.median(voc)];b=pool[voc>np.median(voc)]
  runs=[]
  for seed in (21,37,59):
   rng=np.random.default_rng(seed);p=np.r_[rng.choice(a,10,False),rng.choice(b,10,False)]
   zp=log(tx[p]);sp=log(sx[p]);zt=log(x_test);y=ty[p,1]
   delta=np.median(sp-zp,axis=0)
   unaligned_p=scaler.transform(zp);unaligned_t=scaler.transform(zt)
   aligned_p=scaler.transform(zp+delta);aligned_t=scaler.transform(zt+delta)
   raw0=model.predict(unaligned_p);map0=model.predict(aligned_p)
   pred={'raw_residual':model.predict(unaligned_t)+Ridge(alpha=10).fit(unaligned_p,y-raw0).predict(unaligned_t),
    'paired_map':model.predict(aligned_t),
    'mapped_residual':model.predict(aligned_t)+Ridge(alpha=10).fit(aligned_p,y-map0).predict(aligned_t)}
   high=y_test>=1500
   runs.append({k:{'mae':round(float(abs(y_test-np.clip(v,0,2000)).mean()),2),
    'high_under':round(float(np.maximum(0,y_test[high]-np.clip(v[high],0,2000)).mean()),2)} for k,v in pred.items()})
  dev={k:{metric:round(float(np.median([r[k][metric] for r in runs])),2) for metric in ('mae','high_under')} for k in runs[0]}
  devices[uid]=dev;print(name,uid,dev,flush=True)
 methods=('raw_residual','paired_map','mapped_residual')
 mean={k:{m:round(float(np.mean([v[k][m] for v in devices.values()])),2) for m in ('mae','high_under')} for k in methods}
 best=min(methods[:-1],key=lambda k:mean[k]['mae'])
 candidate=mean['mapped_residual']
 passed=candidate['mae']<=.9*mean[best]['mae'] and candidate['high_under']<=mean[best]['high_under'] and all(v['mapped_residual']['mae']<=1.1*min(v[k]['mae'] for k in methods[:-1]) for v in devices.values())
 out['datasets'][name]={'devices':devices,'mean':mean,'best_baseline':best,'pass':passed}
 print('SUMMARY',name,mean,'best',best,'PASS',passed,flush=True)
out['pass']=all(v['pass'] for v in out['datasets'].values())
print('GLOBAL PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
