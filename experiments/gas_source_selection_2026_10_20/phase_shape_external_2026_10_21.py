"""Frozen external test of heater-response shape for new CO hardware.

Motivation fixed after observing UCI487 (14 sensors): subtraction of heater
phase log responses followed by ExtraTrees was promising. Two DIFFERENT
datasets with four sensor pixels x sixteen phases (64 measurements) are now
tested without seeing this exact representation's results:

6821340: for each A/B/C target, train on all 700 official training recipes
from the other two physical devices; evaluate target's 200 official heldout
recipes. 15423482: for each of seven target devices, train on source six at
state<=100 (100 recipes), test target at state>=160 (40 recipes). Shared
recipe readings from source devices count as one gas exposure. The input is
signed_log(sensor reading) only; no humidity/CO target labels as features.
Shape subtracts each pixel's median across its 16 temperature-phase samples;
raw retains 64 signed-log readings. ExtraTrees(150,leaf2,max_features=1)
and raw Ridge(10) are fixed. Strong baselines: raw ET, raw Ridge, their
fixed 50/50 mean. Pool device MAE separately per dataset; high CO>=1500 ppb
positive underestimation. Success: shape ET is at least 10% better than the
BEST raw baseline in BOTH datasets, no target device worsens >10% against
that dataset's best raw baseline on that device, highCO underestimation does
not worsen on either dataset. Target labels never enter training. This is
not a new blinded dataset because both corpora were used in earlier projects.
"""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

R=Path(__file__).parent
sources=[('6821340','exposures_6821340.npz',['A','B','C']),('15423482_first','exposures_15423482_first.npz',['54118801','54120037','54120457','54156953','84746449','84748697','84749569'])]
out={'protocol':__doc__,'datasets':{}}
def transform(x,shape):
 x=np.sign(x)*np.log1p(abs(x));assert x.shape[1]==64
 if shape:x=x.reshape(-1,4,16);x=x-np.median(x,axis=2,keepdims=True);x=x.reshape(-1,64)
 return x
def maker(name):
 return (ExtraTreesRegressor(n_estimators=150,min_samples_leaf=2,max_features=1.,random_state=21,n_jobs=1) if name=='et' else make_pipeline(StandardScaler(),Ridge(alpha=10)))
for name,filename,devices in sources:
 d=np.load(R/filename)
 rows={}
 for device in devices:
  def load(dev):
   if name=='6821340':
    return (d[f'{dev}_train_x'],d[f'{dev}_train_y'][:,1],d[f'{dev}_test_x'],d[f'{dev}_test_y'][:,1])
   x=d[f'{dev}_x'];info=d[f'{dev}_y'];return x[info[:,0]<=100],info[info[:,0]<=100,1],x[info[:,0]>=160],info[info[:,0]>=160,1]
  train_x=np.concatenate([load(dev)[0] for dev in devices if dev!=device]);train_y=np.concatenate([load(dev)[1] for dev in devices if dev!=device])
  test_x,test_y=load(device)[2:]
  pred={}
  for key,shape,algo in [('raw_et',False,'et'),('raw_ridge',False,'ridge'),('shape_et',True,'et')]:
   model=maker(algo).fit(transform(train_x,shape),train_y)
   pred[key]=np.clip(model.predict(transform(test_x,shape)),0,2000)
  pred['raw_half']=(pred['raw_et']+pred['raw_ridge'])/2
  hard=test_y>=1500
  row={'mae':{k:round(float(abs(test_y-p).mean()),2) for k,p in pred.items()},
    'high_under':{k:round(float(np.maximum(0,test_y[hard]-p[hard]).mean()),2) for k,p in pred.items()}}
  rows[device]=row;print(name,device,row,flush=True)
 methods=('raw_et','raw_ridge','raw_half','shape_et')
 means={k:{metric:round(float(np.mean([v[metric][k] for v in rows.values()])),2) for metric in ('mae','high_under')} for k in methods}
 baseline=min(methods[:-1],key=lambda k:means[k]['mae'])
 shape=means['shape_et'];best=means[baseline]
 success=(shape['mae']<=.9*best['mae'] and shape['high_under']<=best['high_under'] and
   all(v['mae']['shape_et']<=1.1*min(v['mae'][k] for k in methods[:-1]) for v in rows.values()))
 out['datasets'][name]={'devices':rows,'mean':means,'best_raw_method':baseline,'pass':success}
 print('SUMMARY',name,means,'baseline',baseline,'PASS',success,flush=True)
out['pass']=all(v['pass'] for v in out['datasets'].values())
print('GLOBAL PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
