"""Posthoc analysis: select A source scope using 20-pilot LOOCV after correction."""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
d=np.load('exposures_6821340.npz');x=d['A_train_x'];m=d['A_train_y'];models={}
for name,mask in [('ordinary',m[:,0]<=500),('all',np.ones(len(m),bool))]:
 sc=StandardScaler().fit(x[mask]);model=Ridge(alpha=10).fit(sc.transform(x[mask]),m[mask,1]);models[name]=(sc,model)
output={}
for device in ['B','C']:
 train=d[f'{device}_train_y'];xt=d[f'{device}_train_x'];test=d[f'{device}_test_y'];xe=d[f'{device}_test_x']
 normal=np.flatnonzero(train[:,0]<=500);strong=np.flatnonzero((train[:,0]>500)&(train[:,0]<=900))
 mask=test[:,0]<=900;runs=[]
 for seed in range(50):
  rng=np.random.default_rng(20261020+seed);pilot=np.r_[rng.choice(normal,10,replace=False),rng.choice(strong,10,replace=False)]
  cv={};true={}
  for key,(sc,model) in models.items():
   zp=sc.transform(xt[pilot]);ztest=sc.transform(xe[mask]);res=train[pilot,1]-model.predict(zp)
   loo=[]
   for i in range(20):
    tr=np.arange(20)!=i;v=Ridge(alpha=10).fit(zp[tr],res[tr]).predict(zp[i:i+1])[0]
    loo.append(abs(res[i]-v))
   cv[key]=float(np.mean(loo));corr=Ridge(alpha=10).fit(zp,res)
   pred=model.predict(ztest)+corr.predict(ztest)
   true[key]=float(mean_absolute_error(test[mask,1],pred))
  choice=min(cv,key=cv.get);runs.append({'choice':choice,'cv':cv,'test':true,'chosen_test_mae':true[choice]})
 output[device]={'selected_counts':{k:sum(row['choice']==k for row in runs) for k in models},'median_test_mae':{k:round(float(np.median([row['test'][k] for row in runs])),2) for k in models},'selected_median':round(float(np.median([row['chosen_test_mae'] for row in runs])),2),'selected_worse_than_best_fixed_by_10pct':sum(row['chosen_test_mae']>1.1*min(row['test'].values()) for row in runs),'runs':runs}
 print(device,json.dumps({k:v for k,v in output[device].items() if k!='runs'},indent=2),flush=True)
json.dump(output,open('post_calibration_source_cv_6821340_results.json','w'),indent=2)
