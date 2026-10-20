"""Exploratory CO regression with nuisance-covariance penalty.

All hyperparameter choices use source batches 3 and 4 only; later batches are
evaluated as a diagnostic, not an unseen confirmatory test.
"""
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

d=np.load('exposures_4264224.npz');results={}
lam_grid=[0,0.01,0.1,1,10,100]

def fit(x,y,nuisance,idx,lam,alpha=10):
    sx=StandardScaler().fit(x[idx]); xx=sx.transform(x[idx]); sy=StandardScaler().fit(nuisance[idx]);zn=sy.transform(nuisance[idx]);
    yy=y[idx]-y[idx].mean();
    # Remove CO correlation from nuisance in train to avoid penalizing target
    yy_scale=np.std(yy); z_res=zn - (yy[:,None]/yy_scale)*(yy[:,None]*zn).mean(0)[None,:]/yy_scale
    c=(xx.T@z_res)/len(idx)
    a=(xx.T@xx + alpha*np.eye(xx.shape[1]))/len(idx)+lam*(c@c.T)
    coef=np.linalg.solve(a,xx.T@yy/len(idx))
    return lambda new: sx.transform(new)@coef+y[idx].mean()

for sensor in ['as_mlv','as_mlv_p2']:
 x=d[sensor+'_x'];m=d[sensor+'_y'];b=m[:,0].astype(int);y=m[:,2];z=m[:,[3,4,5]]
 val={}
 for lam in lam_grid:
  scores=[]
  for tr,te in [(3,4),(4,3)]:
   f=fit(x,y,z,np.flatnonzero(b==tr),lam)
   test=b==te;scores.append(mean_absolute_error(y[test],f(x[test])))
  val[lam]=float(np.mean(scores))
 best=min(lam_grid,key=lambda l:val[l]);train=np.flatnonzero(np.isin(b,[3,4]))
 fs={str(l):fit(x,y,z,train,l) for l in [0,best,1,10]}
 scores={}
 for batch in [5,6,7,8,10,11,12]:
  ind=b==batch;scores[batch]={name:round(float(mean_absolute_error(y[ind],f(x[ind]))),2) for name,f in fs.items()}
 results[sensor]={'source_cv':{str(l):round(v,2) for l,v in val.items()},'selected_lambda':best,'targets':scores}
print(json.dumps(results,indent=2));json.dump(results,open('interference_invariance_pilot_results.json','w'),indent=2)
