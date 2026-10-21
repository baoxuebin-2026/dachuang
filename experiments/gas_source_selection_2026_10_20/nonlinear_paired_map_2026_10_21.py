"""Frozen nonlinear paired-alignment + target calibration experiment.

Data, source ranges, 20 pilots, source->target paired signal alignment from
paired_response_map_pilot_2026_10_21.py. Source model now fixed HGB(150,
leaf15,minleaf10,l2=10) on source-standardized signed-log response. Fit
20-label Ridge(alpha10) residuals on raw or globally mapped target features.
Baselines: raw HGB residual and previously measured raw Ridge residual,
recomputed here with identical pilot choices. Map-only HGB diagnostic.
Three seeds 21/37/59 share one held-out set; report per-device median.
Success: mapped HGB residual >=10% lower MAE than best raw residual method
on BOTH 6821340 and 15423482, no target device worse >10%, and high CO
>=1500 ppb underestimation no worse on each dataset. The two data sources
have already guided method design, so success still requires external data.
"""
from pathlib import Path
import numpy as np,json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
R=Path(__file__).parent;out={'protocol':__doc__,'datasets':{}}
log=lambda x:np.sign(x)*np.log1p(abs(x))
for dataset,file,source,targets in [('6821340','exposures_6821340.npz','A',['B','C']),('15423482','exposures_15423482_first.npz','54118801',['54120037','54120457','54156953','84746449','84748697','84749569'])]:
 data=np.load(R/file)
 def get(unit):
  if dataset=='6821340':return data[unit+'_train_x'],data[unit+'_train_y'],data[unit+'_test_x'],data[unit+'_test_y']
  return data[unit+'_x'],data[unit+'_y'],None,None
 sx,sy,_,_=get(source)
 source_train=np.arange(500) if dataset=='6821340' else np.where(sy[:,0]<=100)[0]
 pool=np.arange(500,700) if dataset=='6821340' else np.where((sy[:,0]>100)&(sy[:,0]<160))[0]
 sc=StandardScaler().fit(log(sx[source_train]));src=sc.transform(log(sx[source_train]))
 model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=10,random_state=21).fit(src,sy[source_train,1])
 ridge=Ridge(alpha=10).fit(src,sy[source_train,1])
 devs={}
 for uid in targets:
  tx,ty,te_x,te_y=get(uid)
  test_x=te_x if dataset=='6821340' else tx[ty[:,0]>=160]
  test_y=te_y[:,1] if dataset=='6821340' else ty[ty[:,0]>=160,1]
  voc=ty[pool,4 if dataset=='6821340' else 2];low=pool[voc<=np.median(voc)];high=pool[voc>np.median(voc)]
  runs=[]
  for seed in (21,37,59):
   rng=np.random.default_rng(seed);p=np.r_[rng.choice(low,10,False),rng.choice(high,10,False)]
   raw_p=log(tx[p]);raw_t=log(test_x);delta=np.median(log(sx[p])-raw_p,axis=0)
   zp=sc.transform(raw_p);zt=sc.transform(raw_t);mp=sc.transform(raw_p+delta);mt=sc.transform(raw_t+delta);y=ty[p,1]
   def correction(base,a,b):return base.predict(b)+Ridge(alpha=10).fit(a,y-base.predict(a)).predict(b)
   predictions={'ridge_residual':correction(ridge,zp,zt),'hgb_residual':correction(model,zp,zt),
    'hgb_map_only':model.predict(mt),'hgb_map_residual':correction(model,mp,mt)}
   mask=test_y>=1500
   runs.append({name:{'mae':round(float(abs(test_y-np.clip(v,0,2000)).mean()),2),
    'high_under':round(float(np.maximum(0,test_y[mask]-np.clip(v[mask],0,2000)).mean()),2)} for name,v in predictions.items()})
  devs[uid]={k:{metric:round(float(np.median([r[k][metric] for r in runs])),2) for metric in ('mae','high_under')} for k in runs[0]}
  print(dataset,uid,devs[uid],flush=True)
 methods=('ridge_residual','hgb_residual','hgb_map_only','hgb_map_residual')
 avg={k:{metric:round(float(np.mean([d[k][metric] for d in devs.values()])),2) for metric in ('mae','high_under')} for k in methods}
 best=min(methods[:2],key=lambda k:avg[k]['mae']);cand=avg['hgb_map_residual']
 success=cand['mae']<=.9*avg[best]['mae'] and cand['high_under']<=avg[best]['high_under'] and all(d['hgb_map_residual']['mae']<=1.1*min(d[k]['mae'] for k in methods[:2]) for d in devs.values())
 out['datasets'][dataset]={'devices':devs,'mean':avg,'best_baseline':best,'pass':success}
 print('SUMMARY',dataset,avg,'BEST',best,'PASS',success,flush=True)
out['pass']=all(d['pass'] for d in out['datasets'].values())
print('GLOBAL PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
