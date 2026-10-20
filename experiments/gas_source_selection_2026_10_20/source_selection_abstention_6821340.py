"""Posthoc uncertainty gate for 20-pilot source selection.

Criterion fixed to conventional paired 2-SE threshold; deferral requests more
calibrants and makes no claim of a selected model's test performance.
"""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
d=np.load('exposures_6821340.npz');x=d['A_train_x'];m=d['A_train_y'];models={}
for k,mask in [('ordinary',m[:,0]<=500),('all',np.ones(len(m),bool))]:
 sc=StandardScaler().fit(x[mask]);model=Ridge(alpha=10).fit(sc.transform(x[mask]),m[mask,1]);models[k]=(sc,model)
result={}
for device in ['B','C']:
 t=d[f'{device}_train_y'];z=d[f'{device}_train_x'];test=d[f'{device}_test_y'];x_test=d[f'{device}_test_x']
 low=np.flatnonzero(t[:,0]<=500);high=np.flatnonzero((t[:,0]>500)&(t[:,0]<=900));valid=test[:,0]<=900
 run=[]
 for seed in range(50):
  rng=np.random.default_rng(seed+20261020);p=np.r_[rng.choice(low,10,False),rng.choice(high,10,False)]
  losses={};test_err={}
  for key,(sc,f) in models.items():
   pp=sc.transform(z[p]);e=t[p,1]-f.predict(pp)
   losses[key]=np.array([abs(e[i]-Ridge(alpha=10).fit(pp[np.arange(20)!=i],e[np.arange(20)!=i]).predict(pp[i:i+1])[0]) for i in range(20)])
   corr=Ridge(alpha=10).fit(pp,e);test_err[key]=float(mean_absolute_error(test[valid,1],f.predict(sc.transform(x_test[valid]))+corr.predict(sc.transform(x_test[valid]))))
  diff=losses['ordinary']-losses['all'];delta=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(20))
  choice='all' if delta>2*se else 'ordinary' if delta<(-2*se) else 'defer'
  run.append({'choice':choice,'pilot_difference':delta,'pilot_se':se,'test_mae':test_err,'wrong_if_decided':bool(choice!='defer' and test_err[choice]>test_err['ordinary' if choice=='all' else 'all'])})
 result[device]={'decisions':{k:sum(v['choice']==k for v in run) for k in ['ordinary','all','defer']},'incorrect_decisions':sum(v['wrong_if_decided'] for v in run),'runs':run}
 print(device,{k:v for k,v in result[device].items() if k!='runs'})
json.dump(result,open('source_selection_abstention_6821340_results.json','w'),indent=2)
