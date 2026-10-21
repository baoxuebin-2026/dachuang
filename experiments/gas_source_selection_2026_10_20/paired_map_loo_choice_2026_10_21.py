"""Development-only gate, frozen before scoring this candidate gate.

Methods as paired_response_map_pilot_2026_10_21.py. Twenty paired source/
target recipes, stratified 10/10 VOC; seeds 21/37/59. Compare paired-map
and residual-Ridge using leave-one-recipe-out: for each held recipe, rebuild
median source-target response displacement on other 19 or residual Ridge on
other 19. Paired errors d = abs(residual error)-abs(map error). Choose paired
map only if mean(d)>one standard error of d; else default to residual Ridge.
Then train chosen method on all 20 and evaluate official / state test.
Predeclared development criterion: map selected for BOTH B and C in >=2/3
seeds, residual selected for >=5 of 6 devices in >=2/3 seeds, and per-source
heldout averaged MAE at least 10% below always-residual on 682 while no
more than 10% above always-residual on 154. Repeated seeds not independent.
This experiment is not blind: gate constructed after viewing both datasets'
ungated performances. Any success needs a third untouched paired CO dataset.
"""
from pathlib import Path
import numpy as np,json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

R=Path(__file__).parent;output={'protocol':__doc__,'datasets':{}}
for dataset,filename,source,targets in [
 ('6821340','exposures_6821340.npz','A',['B','C']),
 ('15423482','exposures_15423482_first.npz','54118801',['54120037','54120457','54156953','84746449','84748697','84749569'])]:
 data=np.load(R/filename)
 def get(uid):
  if dataset=='6821340':return data[f'{uid}_train_x'],data[f'{uid}_train_y'],data[f'{uid}_test_x'],data[f'{uid}_test_y']
  return data[uid+'_x'],data[uid+'_y'],None,None
 log=lambda a:np.sign(a)*np.log1p(abs(a))
 sx,sy,_,_=get(source)
 train=np.arange(500) if dataset=='6821340' else np.flatnonzero(sy[:,0]<=100)
 pool=np.arange(500,700) if dataset=='6821340' else np.flatnonzero((sy[:,0]>100)&(sy[:,0]<160))
 scale=StandardScaler().fit(log(sx[train]));base=Ridge(alpha=10).fit(scale.transform(log(sx[train])),sy[train,1])
 target_results={}
 for uid in targets:
  tx,ty,t_x,t_y=get(uid)
  test_x=t_x if dataset=='6821340' else tx[ty[:,0]>=160]
  test_y=t_y[:,1] if dataset=='6821340' else ty[ty[:,0]>=160,1]
  voc=ty[pool,4 if dataset=='6821340' else 2];a=pool[voc<=np.median(voc)];b=pool[voc>np.median(voc)]
  runs=[]
  for seed in (21,37,59):
   rng=np.random.default_rng(seed);p=np.r_[rng.choice(a,10,False),rng.choice(b,10,False)]
   z=log(tx[p]);s=log(sx[p]);zt=log(test_x);y=ty[p,1]
   zsrc=scale.transform(z);p0=base.predict(zsrc)
   lmap=[];lres=[]
   for held in range(20):
    tr=np.arange(20)!=held
    m=base.predict(scale.transform(z[held:held+1]+np.median(s[tr]-z[tr],axis=0)))[0]
    r=base.predict(zsrc[held:held+1])[0]+Ridge(alpha=10).fit(zsrc[tr],y[tr]-p0[tr]).predict(zsrc[held:held+1])[0]
    lmap.append(abs(y[held]-m));lres.append(abs(y[held]-r))
   delta=np.asarray(lres)-np.asarray(lmap)
   se=delta.std(ddof=1)/np.sqrt(20)
   choice='paired_map' if delta.mean()>se else 'residual'
   mapped=base.predict(scale.transform(zt+np.median(s-z,axis=0)))
   corrected=base.predict(scale.transform(zt))+Ridge(alpha=10).fit(zsrc,y-p0).predict(scale.transform(zt))
   scores={name:round(float(abs(test_y-np.clip(pred,0,2000)).mean()),2) for name,pred in [('paired_map',mapped),('residual',corrected)]}
   runs.append({'seed':seed,'choice':choice,'loo_delta':round(float(delta.mean()),2),'loo_se':round(float(se),2),'test_mae':scores})
  target_results[uid]=runs
  print(dataset,uid,[{'choice':r['choice'],'loo':r['loo_delta'],'mae':r['test_mae']} for r in runs],flush=True)
 chosen=np.array([r['test_mae'][r['choice']] for runs in target_results.values() for r in runs]);baseline=np.array([r['test_mae']['residual'] for runs in target_results.values() for r in runs])
 stats={'selected_map_counts':{k:sum(r['choice']=='paired_map' for r in runs) for k,runs in target_results.items()},
  'chosen_mae':round(float(chosen.mean()),2),'residual_mae':round(float(baseline.mean()),2)}
 output['datasets'][dataset]={'targets':target_results,'summary':stats};print('SUMMARY',dataset,stats,flush=True)
A=output['datasets']['6821340']['summary'];B=output['datasets']['15423482']['summary']
output['pass']=(all(n>=2 for n in A['selected_map_counts'].values()) and
 sum(n<=1 for n in B['selected_map_counts'].values())>=5 and
 A['chosen_mae']<=.9*A['residual_mae'] and B['chosen_mae']<=1.1*B['residual_mae'])
print('DEVELOPMENT PASS',output['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(output,indent=2))
