"""Exploratory choice of no, scalar, or response-dependent calibration.

All model selection is restricted to the first 20 labeled exposures of each
target lab/batch. LOOCV uses exposure as unit. Later exposure labels enter
the code only to calculate outcome, never selection. This was conceived after
examining earlier batch results; external performance remains exploratory.
"""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def choose(z,e):
 n=len(e);loo={'none':np.zeros(n),'offset':(sum(e)-e)/(n-1)}
 p=[]
 for i in range(n):
  train=np.arange(n)!=i
  p.append(Ridge(alpha=10).fit(z[train],e[train]).predict(z[i:i+1])[0])
 loo['residual']=np.array(p)
 # Ordered complexity, requiring the candidate to improve pilot MAE by
 # more than one paired standard error over the currently selected method.
 best='none'
 for candidate in ['offset','residual']:
  d=np.abs(e-loo[best])-np.abs(e-loo[candidate])
  if d.mean()>d.std(ddof=1)/np.sqrt(n):best=candidate
 return best,{k:round(float(np.abs(e-v).mean()),2) for k,v in loo.items()}

result={}
for dataset in ['4264224','4593853']:
 d=np.load('exposures_'+dataset+'.npz');out={}
 names=['as_mlv','as_mlv_p2'] if dataset=='4264224' else ['sensor0','sensor1']
 for sensor in names:
  x=d[sensor+'_x'];m=d[sensor+'_y'];batch=m[:,0 if dataset=='4264224' else 1].astype(int);y=m[:,2]
  source=np.isin(batch,[3,4]) if dataset=='4264224' else batch==1
  sc=StandardScaler().fit(x[source]);z=sc.transform(x)
  f=Ridge(alpha=10).fit(z[source],y[source]);unit={}
  target_batches=[5,6,7,8,10,11,12] if dataset=='4264224' else [2,3,4,5,6]
  for b in target_batches:
   idx=np.flatnonzero(batch==b);pilot=idx[:20];test=idx[20:]
   e=y[pilot]-f.predict(z[pilot]);method,cv=choose(z[pilot],e)
   raw=f.predict(z[test]);offset=raw+e.mean()
   ridge=raw+Ridge(alpha=10).fit(z[pilot],e).predict(z[test])
   predictions={'none':raw,'offset':offset,'residual':ridge}
   unit[str(b)]={'n_test':len(test),'choice':method,'pilot_loo_mae':cv,
    'test_mae':{key:round(float(mean_absolute_error(y[test],val)),2) for key,val in predictions.items()},
    'chosen_mae':round(float(mean_absolute_error(y[test],predictions[method])),2)}
  out[sensor]=unit
 result[dataset]=out
print(json.dumps(result,indent=2));json.dump(result,open('calibration_complexity_selection_results.json','w'),indent=2)
