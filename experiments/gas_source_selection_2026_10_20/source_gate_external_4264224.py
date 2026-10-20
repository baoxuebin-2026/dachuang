"""Frozen alternative dataset stress test for post-calibration model-scope gate."""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
d=np.load('exposures_4264224.npz');out={}
for sensor in ['as_mlv','as_mlv_p2']:
 x=d[sensor+'_x'];m=d[sensor+'_y'];batch=m[:,0].astype(int);y=m[:,2];sources={}
 for key,mask in [('batch3',batch==3),('batch3_4',np.isin(batch,[3,4]))]:
  sc=StandardScaler().fit(x[mask]);f=Ridge(alpha=10).fit(sc.transform(x[mask]),y[mask]);sources[key]=(sc,f)
 res={}
 for b in [5,6,7,8,10,11,12]:
  idx=np.flatnonzero(batch==b);pilot=idx[:20];test=idx[20:];loss={};true={}
  for key,(sc,f) in sources.items():
   zp=sc.transform(x[pilot]);zt=sc.transform(x[test]);e=y[pilot]-f.predict(zp)
   loss[key]=np.array([abs(e[i]-Ridge(alpha=10).fit(zp[np.arange(20)!=i],e[np.arange(20)!=i]).predict(zp[i:i+1])[0]) for i in range(20)])
   pred=f.predict(zt)+Ridge(alpha=10).fit(zp,e).predict(zt)
   true[key]=round(float(mean_absolute_error(y[test],pred)),2)
  diff=loss['batch3']-loss['batch3_4'];delta=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(20))
  choice='batch3_4' if delta>2*se else 'batch3' if delta<(-2*se) else 'defer'
  other='batch3' if choice=='batch3_4' else 'batch3_4'
  res[str(b)]={'n_test':len(test),'choice':choice,'difference_ppb':round(delta,2),'se_ppb':round(se,2),'test_mae':true,'wrong_by_10pct':bool(choice!='defer' and true[choice]>1.1*true[other])}
 out[sensor]=res
 print(sensor,res,flush=True)
json.dump(out,open('source_gate_external_4264224_results.json','w'),indent=2)
