"""Run preregistered 20-exposure calibration allocation comparison."""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

d=np.load('exposures_6821340.npz')
src_m=d['A_train_y'];src_x=d['A_train_x'];src_ids=src_m[:,0]
src_low=src_ids<=500
sc=StandardScaler().fit(src_x[src_low]);f=Ridge(alpha=10).fit(sc.transform(src_x[src_low]),src_m[src_low,1])
sc_all=StandardScaler().fit(src_x);f_all=Ridge(alpha=10).fit(sc_all.transform(src_x),src_m[:,1])
out={}
for sensor in ['B','C']:
 train_m=d[f'{sensor}_train_y'];test_m=d[f'{sensor}_test_y'];train_x=sc.transform(d[f'{sensor}_train_x']);test_x=sc.transform(d[f'{sensor}_test_x'])
 train_ids=train_m[:,0];test_ids=test_m[:,0]
 low=np.flatnonzero(train_ids<=500);high=np.flatnonzero((train_ids>500)&(train_ids<=900));all_pool=np.r_[low,high]
 test_groups={'ordinary':test_ids<=500,'strong':(test_ids>500)&(test_ids<=900),'all':test_ids<=900}
 base=f.predict(test_x);strong_source=f_all.predict(sc_all.transform(d[f'{sensor}_test_x']))
 benchmarks={key:{'source_low':round(float(mean_absolute_error(test_m[mask,1],base[mask])),2),'source_all':round(float(mean_absolute_error(test_m[mask,1],strong_source[mask])),2)} for key,mask in test_groups.items()}
 scheme_scores={k:[] for k in ['ordinary20','strong20','balanced10_10','random20']}
 for seed in range(50):
  rng=np.random.default_rng(seed+20261020)
  chosen={
   'ordinary20':rng.choice(low,20,replace=False),
   'strong20':rng.choice(high,20,replace=False),
   'balanced10_10':np.r_[rng.choice(low,10,replace=False),rng.choice(high,10,replace=False)],
   'random20':rng.choice(all_pool,20,replace=False),
  }
  for key,idx in chosen.items():
   residual=train_m[idx,1]-f.predict(train_x[idx]);g=Ridge(alpha=10).fit(train_x[idx],residual)
   pred=base+g.predict(test_x)
   errors={group:float(mean_absolute_error(test_m[mask,1],pred[mask])) for group,mask in test_groups.items()}
   errors['worst_group']=max(errors['ordinary'],errors['strong'])
   scheme_scores[key].append(errors)
 summary={k:{q:{'median':round(float(np.median([a[q] for a in vals])),2),'min':round(float(min(a[q] for a in vals)),2),'max':round(float(max(a[q] for a in vals)),2)} for q in ['ordinary','strong','all','worst_group']} for k,vals in scheme_scores.items()}
 out[sensor]={'train_pool':{'ordinary':len(low),'strong':len(high)},'test_counts':{k:int(v.sum()) for k,v in test_groups.items()},'benchmarks':benchmarks,'summary':summary,'per_seed':scheme_scores}
 print(sensor,json.dumps({'benchmarks':benchmarks,'summary':summary},indent=2),flush=True)
json.dump(out,open('mixture_budget_allocation_6821340_results.json','w'),indent=2)
