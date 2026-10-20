"""Test frozen conditional VOC-interference penalty on official held-out mixtures."""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

d=np.load('exposures_6821340.npz')
m=d['A_train_y'];r=m[:,0];y=m[:,1]

def fit_penalty(X,m,indices,lam):
 sx=StandardScaler().fit(X[indices]);xx=sx.transform(X[indices]);yy=m[indices,1];yc=yy-yy.mean()
 nuisance=m[indices,2:5] # hydrogen, humidity, VOC
 sn=StandardScaler().fit(nuisance);nn=sn.transform(nuisance)
 z=np.column_stack([np.ones(len(indices)),(yc/yc.std()),nn[:,:2]])
 # Target VOC residual after conditioning on CO, H2, and water.
 v=nn[:,2]-z@np.linalg.lstsq(z,nn[:,2],rcond=None)[0]
 c=xx.T@v/len(indices)
 mat=(xx.T@xx+10*np.eye(xx.shape[1]))/len(indices)+lam*np.outer(c,c)
 beta=np.linalg.solve(mat,xx.T@yc/len(indices))
 return lambda new:sx.transform(new)@beta+yy.mean()

grid=[0,.01,.1,1,10,100]
crossval={}
for lam in grid:
 scores=[]
 for lo,hi in [((r<=200),(r>200)&(r<=500)),(((r>200)&(r<=500)),r<=200)]:
  f=fit_penalty(d['A_train_x'],m,np.flatnonzero(lo),lam)
  scores.append(mean_absolute_error(y[hi],f(d['A_train_x'][hi])))
 crossval[str(lam)]=[round(float(s),2) for s in scores]
selected=min(grid,key=lambda k:np.mean(crossval[str(k)]))
source=(r<=500);idx=np.flatnonzero(source)
candidate=fit_penalty(d['A_train_x'],m,idx,selected)
ridge=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(d['A_train_x'][idx],y[idx])
svr=make_pipeline(StandardScaler(),TransformedTargetRegressor(regressor=SVR(C=10,epsilon=.1,gamma='scale'),transformer=StandardScaler())).fit(d['A_train_x'][idx],y[idx])
forest=RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=20261020,n_jobs=2).fit(d['A_train_x'][idx],y[idx])
models={'ridge':ridge.predict,'nuisance_penalty':candidate,'rbf_svr':svr.predict,'random_forest':forest.predict}
result={'source_cv':crossval,'lambda':selected,'source_n':len(idx),'scores':{}}
for sensor in ['A','B','C']:
 for partition in ['test']:
  X=d[f'{sensor}_{partition}_x'];label=d[f'{sensor}_{partition}_y'];rr=label[:,0]
  for group,mask in [('low_mid',rr<=500),('strong_voc',(rr>500)&(rr<=900))]:
   result['scores'][f'{sensor}_{partition}_{group}']={'n':int(mask.sum()),'mae_ppb':{key:round(float(mean_absolute_error(label[mask,1],model(X[mask]))),2) for key,model in models.items()}}
print(json.dumps(result,indent=2));json.dump(result,open('interference_direction_6821340_results.json','w'),indent=2)
