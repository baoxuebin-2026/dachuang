"""Frozen response-distance error audit, independent 3/7-device datasets.

Protocol/source/test and ExtraTrees identical to device_disagreement. For
each unit, scale 64 signed-log responses by training median and robust
interquartile span (floor 0.1), use Euclidean distance to fifth nearest
training recipe; aggregate seven/three unit distances by median. Rank official
heldout exposures by distance. Primary score: top 25% queue captures at
least 40% of top 20% absolute error from the cross-device mean prediction,
and has >=1.5x mean error vs bottom 75%, on BOTH datasets. No label-based
threshold/parameter selection. Compare against device-disagreement ranking
using same strong mean ensemble prediction. Success justifies next blind test
and does not imply novel mathematics, guaranteed CO detection, or transfer.
"""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors

R=Path(__file__).parent
settings=[('6821340','exposures_6821340.npz',['A','B','C']),('15423482_first','exposures_15423482_first.npz',['54118801','54120037','54120457','54156953','84746449','84748697','84749569'])]
out={'protocol':__doc__,'datasets':{}}
for key,filename,devices in settings:
 d=np.load(R/filename);predictions=[];distance=[];truth=None
 for unit in devices:
  if key=='6821340':
   train=d[f'{unit}_train_x'];yt=d[f'{unit}_train_y'][:,1]
   test=d[f'{unit}_test_x'];y=d[f'{unit}_test_y'][:,1]
  else:
   X=d[f'{unit}_x'];info=d[f'{unit}_y'];mask_train=info[:,0]<=100;mask_test=info[:,0]>=160
   train=X[mask_train];yt=info[mask_train,1];test=X[mask_test];y=info[mask_test,1]
  if truth is None:truth=y
  else:assert np.allclose(truth,y)
  slog=lambda v:np.sign(v)*np.log1p(abs(v))
  train=slog(train);test=slog(test)
  model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=.8,random_state=21,n_jobs=1)
  predictions.append(np.clip(model.fit(train,yt).predict(test),0,max(yt)))
  center=np.median(train,axis=0);span=np.maximum(np.percentile(train,75,axis=0)-np.percentile(train,25,axis=0),.1)
  knn=NearestNeighbors(n_neighbors=5).fit((train-center)/span)
  distance.append(knn.kneighbors((test-center)/span)[0][:,-1])
 P=np.stack(predictions);D=np.median(distance,axis=0)
 error=abs(P.mean(axis=0)-truth);n=len(truth)
 def evaluate(rank):
  q=np.argsort(-rank)[:int(np.ceil(.25*n))];hard=np.argsort(-error)[:int(np.ceil(.2*n))]
  flag=np.zeros(n,dtype=bool);flag[q]=True
  return {'hard_capture':round(float(np.isin(hard,q).mean()),3),'enrichment':round(float(error[flag].mean()/error[~flag].mean()),3)}
 result={'n_test':n,'ensemble_mean_mae':round(float(error.mean()),3),'distance':evaluate(D),
   'device_disagreement':evaluate(np.median(abs(P-np.median(P,axis=0)),axis=0))}
 out['datasets'][key]=result;print(key,result,flush=True)
out['pass']=all(r['distance']['hard_capture']>=.4 and r['distance']['enrichment']>=1.5 for r in out['datasets'].values())
print('PASS',out['pass'],flush=True)
(Path(__file__).with_suffix('.json')).write_text(json.dumps(out,indent=2))
