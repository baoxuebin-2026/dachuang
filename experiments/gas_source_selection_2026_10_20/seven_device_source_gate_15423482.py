"""Frozen seven-device source-scope gate pressure test (first calibration)."""
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
d=np.load('exposures_15423482_first.npz');uids=sorted(set(k[:-2] for k in d if k.endswith('_x')))
source=uids[0];x0=d[source+'_x'];m0=d[source+'_y'];source_mask=m0[:,0]<=100
normal_source=source_mask&(m0[:,2]<=np.median(m0[source_mask,2]));models={}
for key,mask in [('ordinary',normal_source),('all',source_mask)]:
 sc=StandardScaler().fit(x0[mask]);model=Ridge(alpha=10).fit(sc.transform(x0[mask]),m0[mask,1]);models[key]=(sc,model)
result={'source_device':source,'source_counts':{'ordinary':int(normal_source.sum()),'all':int(source_mask.sum())},'targets':{}}
for uid in uids[1:]:
 x=d[uid+'_x'];m=d[uid+'_y'];ids=m[:,0];cand=(ids>100)&(ids<160);test=ids>=160
 median=np.median(m[cand,2]);low=np.flatnonzero(cand&(m[:,2]<=median));high=np.flatnonzero(cand&(m[:,2]>median))
 runs=[]
 for seed in range(50):
  rng=np.random.default_rng(seed+20261020);pilot=np.r_[rng.choice(low,10,False),rng.choice(high,10,False)]
  loss={};test_error={}
  for scope,(sc,f) in models.items():
   zp=sc.transform(x[pilot]);zt=sc.transform(x[test]);e=m[pilot,1]-f.predict(zp)
   loss[scope]=np.array([abs(e[i]-Ridge(alpha=10).fit(zp[np.arange(20)!=i],e[np.arange(20)!=i]).predict(zp[i:i+1])[0]) for i in range(20)])
   model=Ridge(alpha=10).fit(zp,e)
   test_error[scope]=float(mean_absolute_error(m[test,1],f.predict(zt)+model.predict(zt)))
  diff=loss['ordinary']-loss['all'];z=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(20))
  decision='all' if z>2*se else 'ordinary' if z<(-2*se) else 'defer'
  other='ordinary' if decision=='all' else 'all'
  runs.append({'decision':decision,'pilot_difference_ppb':z,'pilot_se_ppb':se,'test_mae':test_error,'severe_wrong':bool(decision!='defer' and test_error[decision]>1.1*test_error[other])})
 summary={'decisions':{k:sum(a['decision']==k for a in runs) for k in ['ordinary','all','defer']},'severely_wrong':sum(a['severe_wrong'] for a in runs),'median_test_mae':{k:round(float(np.median([a['test_mae'][k] for a in runs])),2) for k in ['ordinary','all']}}
 result['targets'][uid]={'summary':summary,'runs':runs}
 print(uid,summary,flush=True)
json.dump(result,open('seven_device_source_gate_15423482_results.json','w'),indent=2)
