"""Exploratory sequential 10-or-20 exposure output-bias calibration.

Tests always use exposures 21 onwards; this keeps the evaluation set identical
across 10/20 calibration budgets. Threshold constants are conventional, not
fitted to target batches.
"""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
d=np.load('exposures_4264224.npz');result={}
for sensor in ['as_mlv','as_mlv_p2']:
 x=d[sensor+'_x'];m=d[sensor+'_y'];b=m[:,0].astype(int);co=m[:,2]
 source=np.isin(b,[3,4]);sc=StandardScaler().fit(x[source]);z=sc.transform(x)
 f=Ridge(alpha=10).fit(z[source],co[source]);r={}
 for batch in [5,6,7,8,10,11,12]:
  a=np.flatnonzero(b==batch);cal=a[:20];test=a[20:]
  e=co[cal]-f.predict(z[cal]);pred=f.predict(z[test])
  def estimate(n):
   q=e[:n];return float(q.mean()),float(q.std(ddof=1)/np.sqrt(n))
  b5,s5=estimate(5);b10,s10=estimate(10);b20,s20=estimate(20)
  # Stop early only if the bias is clearly nonzero and remains stable
  # in the first and second five exposures.
  early=(abs(b10)>2*s10 and np.sign(b10)==np.sign(b5) and abs(b10-b5)<=np.sqrt(s10*s10+s5*s5))
  n=10 if early else 20
  shift=b10 if early else b20 if abs(b20)>2*s20 else 0.0
  r[str(batch)]={'calibration_exposures':n,'raw_mae':round(float(mean_absolute_error(co[test],pred)),2),
   'sequential_mae':round(float(mean_absolute_error(co[test],pred+shift)),2),
   'fixed20_mae':round(float(mean_absolute_error(co[test],pred+(b20 if abs(b20)>2*s20 else 0))),2),
   'b5':round(b5,1),'b10':round(b10,1),'b20':round(b20,1)}
 result[sensor]=r
print(json.dumps(result,indent=2));json.dump(result,open('sequential_anchor_pilot_results.json','w'),indent=2)
