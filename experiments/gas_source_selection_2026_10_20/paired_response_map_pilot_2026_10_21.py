"""Pre-run frozen paired-response transfer pilot for 20 known CO mixtures.

Source A on 6821340 uses first 500 training mixtures; heldout test is
official 200 mixtures, B/C targets use pilots from source+target train
mixtures 500:700. 15423482 uses device 54118801 source <=100, six other
targets use pilots from states 101:159, test state >=160. Same pilot budget:
10 below/10 above median VOC among available pilot mixtures. Three fixed
seeds (21,37,59) are sensitivity repeats, not new independent datasets.

Source Ridge(10) is trained on signed-log signals and its own train scaler.
Strong target baselines: supervised Ridge(10) on 20 target mixtures, and
source Ridge plus target 20-label residual Ridge(10), the latter identical
to prior work's calibration. Proposed paired map estimates target->source
response correction from the 20 physically matched mixtures. Two FIXED
versions: median per-feature displacement (global) and 5-nearest target
response neighbors' inverse-distance weighted displacements (local). No
test labels or target test feature distribution used to fit. Predeclared
pass for LOCAL map: median MAE across B/C and all 6 154 targets >=10% below
the best target baseline on each dataset, no individual target degrades
>10%, and highCO>=1500 ppb positive underestimation no worse on either.
This local paired map is related to known calibration-transfer techniques;
success would not alone prove mathematical originality.
"""
from pathlib import Path
import numpy as np,json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

R=Path(__file__).parent
out={'protocol':__doc__,'datasets':{}}
for dataset,filename,source,targets in [
 ('6821340','exposures_6821340.npz','A',['B','C']),
 ('15423482','exposures_15423482_first.npz','54118801',['54120037','54120457','54156953','84746449','84748697','84749569'])]:
 d=np.load(R/filename)
 def response(unit):
  if dataset=='6821340':return d[unit+'_train_x'],d[unit+'_train_y'],d[unit+'_test_x'],d[unit+'_test_y']
  return d[unit+'_x'],d[unit+'_y'],None,None
 log=lambda x:np.sign(x)*np.log1p(abs(x))
 xs,ys,xst,yst=response(source)
 source_train=np.arange(500) if dataset=='6821340' else np.flatnonzero(ys[:,0]<=100)
 source_pilot=np.arange(500,700) if dataset=='6821340' else np.flatnonzero((ys[:,0]>100)&(ys[:,0]<160))
 source_scaler=StandardScaler().fit(log(xs[source_train]));model=Ridge(alpha=10).fit(source_scaler.transform(log(xs[source_train])),ys[source_train,1])
 results={}
 for target in targets:
  xt,yt,xtest,ytest=response(target)
  pool=source_pilot
  test_x=xtest if dataset=='6821340' else xt[yt[:,0]>=160]
  test_y=ytest[:,1] if dataset=='6821340' else yt[yt[:,0]>=160,1]
  assert len(ys)==len(yt) and np.allclose(ys[:,1],yt[:,1])
  voc=yt[pool,4 if dataset=='6821340' else 2]
  low=pool[voc<=np.median(voc)];high=pool[voc>np.median(voc)]
  runs=[]
  for seed in (21,37,59):
   rng=np.random.default_rng(seed);pilots=np.r_[rng.choice(low,10,False),rng.choice(high,10,False)]
   tp=log(xt[pilots]);sp=log(xs[pilots]);test=log(test_x)
   plain=model.predict(source_scaler.transform(test));onpilot=model.predict(source_scaler.transform(tp))
   target_scaler=StandardScaler().fit(tp)
   direct=Ridge(alpha=10).fit(target_scaler.transform(tp),yt[pilots,1]).predict(target_scaler.transform(test))
   residual_model=Ridge(alpha=10).fit(source_scaler.transform(tp),yt[pilots,1]-onpilot)
   residual=plain+residual_model.predict(source_scaler.transform(test))
   displacement=sp-tp
   global_map=model.predict(source_scaler.transform(test+np.median(displacement,axis=0)))
   scale=np.maximum(np.percentile(tp,75,axis=0)-np.percentile(tp,25,axis=0),.1)
   near=NearestNeighbors(n_neighbors=5).fit(tp/scale)
   dd,ii=near.kneighbors(test/scale)
   weights=1/np.maximum(dd,1e-6);weights/=weights.sum(axis=1,keepdims=True)
   local_map=model.predict(source_scaler.transform(test+(displacement[ii]*weights[:,:,None]).sum(axis=1)))
   preds={'direct20':direct,'residual20':residual,'global_paired':global_map,'local_paired':local_map}
   metric={k:{'mae':round(float(np.abs(test_y-np.clip(v,0,2000)).mean()),2),
    'high_under':round(float(np.maximum(0,test_y[test_y>=1500]-np.clip(v[test_y>=1500],0,2000)).mean()),2)} for k,v in preds.items()}
   runs.append(metric)
  stats={k:{field:round(float(np.median([r[k][field] for r in runs])),2) for field in ('mae','high_under')} for k in runs[0]}
  results[target]=stats;print(dataset,target,stats,flush=True)
 avg={k:{field:round(float(np.mean([v[k][field] for v in results.values()])),2) for field in ('mae','high_under')} for k in ('direct20','residual20','global_paired','local_paired')}
 best=min(('direct20','residual20'),key=lambda k:avg[k]['mae'])
 success=avg['local_paired']['mae']<=.9*avg[best]['mae'] and avg['local_paired']['high_under']<=avg[best]['high_under'] and all(v['local_paired']['mae']<=1.1*min(v[k]['mae'] for k in ('direct20','residual20')) for v in results.values())
 out['datasets'][dataset]={'per_target':results,'mean':avg,'best_target_baseline':best,'pass':success}
 print('SUMMARY',dataset,avg,'baseline',best,'PASS',success,flush=True)
out['pass']=all(v['pass'] for v in out['datasets'].values())
print('GLOBAL PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
