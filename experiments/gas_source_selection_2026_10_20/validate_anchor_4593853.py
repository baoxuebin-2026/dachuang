"""Run prewritten 20-exposure output-bias gate on Zenodo 4593853.

See 外部验证预先规则_2026-10-20.md for protocol written before this run.
"""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

d=np.load('exposures_4593853.npz');result={}
for name in ['sensor0','sensor1']:
 x=d[name+'_x'];meta=d[name+'_y'];lab=meta[:,1].astype(int);co=meta[:,2]
 train=lab==1;sc=StandardScaler().fit(x[train]);z=sc.transform(x)
 f=Ridge(alpha=10).fit(z[train],co[train]);mu=z[train].mean(0)
 sensor={}
 for batch in [2,3,4,5,6]:
  ind=np.flatnonzero(lab==batch);pilot=ind[:20];test=ind[20:]
  e=co[pilot]-f.predict(z[pilot]);bias=float(e.mean());se=float(e.std(ddof=1)/np.sqrt(len(e)))
  raw=f.predict(z[test]);center=f.predict(z[test]-(z[pilot].mean(0)-mu))
  corrected=raw+bias if abs(bias)>2*se else raw
  residual=raw+Ridge(alpha=10).fit(z[pilot],e).predict(z[test])
  sensor[str(batch)]={'n_train':int(sum(train)),'n_pilot':len(pilot),'n_test':len(test),
   'bias_ppb':round(bias,2),'se_ppb':round(se,2),'gate_activated':bool(abs(bias)>2*se),
   'raw_mae':round(float(mean_absolute_error(co[test],raw)),2),
   'unlabeled_center_mae':round(float(mean_absolute_error(co[test],center)),2),
   'gated_bias_mae':round(float(mean_absolute_error(co[test],corrected)),2),
   'residual_ridge20_mae':round(float(mean_absolute_error(co[test],residual)),2)}
 result[name]=sensor
print(json.dumps(result,ensure_ascii=False,indent=2));json.dump(result,open('validate_anchor_4593853_results.json','w'),indent=2)
