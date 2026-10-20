"""Select source training scope using 20 labeled target-device mixtures."""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

d=np.load('exposures_6821340.npz');m=d['A_train_y'];x=d['A_train_x']
sources={}
for name,mask in [('ordinary',m[:,0]<=500),('all',np.ones(len(m),bool))]:
 scaler=StandardScaler().fit(x[mask]);model=Ridge(alpha=10).fit(scaler.transform(x[mask]),m[mask,1])
 sources[name]=(scaler,model)
result={}
for device in ['B','C']:
 y_train=d[f'{device}_train_y'];x_train=d[f'{device}_train_x'];y_test=d[f'{device}_test_y'];x_test=d[f'{device}_test_x']
 ids=y_train[:,0];normal=np.flatnonzero(ids<=500);severe=np.flatnonzero((ids>500)&(ids<=900))
 test_groups={'ordinary':y_test[:,0]<=500,'strong':(y_test[:,0]>500)&(y_test[:,0]<=900),'all':y_test[:,0]<=900}
 all_runs=[]
 for seed in range(50):
  rng=np.random.default_rng(20261020+seed)
  pilot=np.r_[rng.choice(normal,10,replace=False),rng.choice(severe,10,replace=False)]
  pil_score={};out={}
  for scope,(sc,model) in sources.items():
   zx=sc.transform(x_train[pilot]);zy=sc.transform(x_test)
   pil_score[scope]=float(mean_absolute_error(y_train[pilot,1],model.predict(zx)))
   correction=Ridge(alpha=10).fit(zx,y_train[pilot,1]-model.predict(zx))
   raw=model.predict(zy);corrected=raw+correction.predict(zy)
   out[scope]={name:float(mean_absolute_error(y_test[mask,1],corrected[mask])) for name,mask in test_groups.items()}
   out[scope]['worst']=max(out[scope]['ordinary'],out[scope]['strong'])
  chosen=min(pil_score,key=pil_score.get)
  all_runs.append({'chosen':chosen,'pilot_mae':pil_score,'corrected':out})
 def stat(scope,metric):
  a=[q['corrected'][q['chosen'] if scope=='chosen' else scope][metric] for q in all_runs]
  return {'median':round(float(np.median(a)),2),'min':round(float(min(a)),2),'max':round(float(max(a)),2)}
 summary={key:{metric:stat(key,metric) for metric in ['ordinary','strong','all','worst']} for key in ['ordinary','all','chosen']}
 result[device]={'selection_counts':{scope:sum(v['chosen']==scope for v in all_runs) for scope in sources},'summary':summary,'runs':all_runs}
 print(device,json.dumps({'selection_counts':result[device]['selection_counts'],'summary':summary},indent=2),flush=True)
json.dump(result,open('source_scope_selection_6821340_results.json','w'),indent=2)
