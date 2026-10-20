"""Reduce 2023 first SGP40 calibration to complete mixture exposures."""
import json,numpy as np
from scipy.io import loadmat
D=loadmat('zenodo_15423482_Zenodo_First_Calibration.mat',variable_names=['sensor','target'],simplify_cells=True)
t=D['target'];states=t['states'];co=t['carbonMonoxide'];
voc=sum(t[k] for k in ['acetone','ethanol','ethylAcetate','formaldehyde','toluene'])
all_valid=np.isfinite(states)&np.isfinite(co)&np.isfinite(voc)
rows=np.linspace(60,1380,16,dtype=int);out={}
uids=sorted({name.split('_')[1] for name in D['sensor']})
for uid in uids:
 sensors=[D['sensor'][f'SGP40_{uid}_sensor{p}'] for p in range(4)]
 phases=np.concatenate([s[:,rows] for s in sensors],axis=1)
 X=[];M=[]
 for state in np.unique(states[all_valid]):
  ind=np.flatnonzero(all_valid&(states==state))
  feature=np.nanmedian(phases[ind],axis=0)
  label=[state,np.nanmedian(co[ind]),np.nanmedian(voc[ind]),np.nanmedian(t['hydrogen'][ind]),np.nanmedian(t['water'][ind])]
  if np.all(np.isfinite(feature)) and np.all(np.isfinite(label)):
   X.append(feature);M.append(label)
 out[uid+'_x']=np.array(X);out[uid+'_y']=np.array(M)
 print(uid,len(X),'source',sum(out[uid+'_y'][:,0]<=100),'pilot_pool',sum((out[uid+'_y'][:,0]>100)&(out[uid+'_y'][:,0]<160)),'test',sum(out[uid+'_y'][:,0]>=160),flush=True)
np.savez_compressed('exposures_15423482_first.npz',**out)
json.dump({'source':'https://doi.org/10.5281/zenodo.15423482','file_md5':'0205135ec71cf52c9b33fe1215d61cd5','units':len(uids),'valid_exposures_per_device':{k:int(out[k+'_x'].shape[0]) for k in uids},'phase_rows':rows.tolist(),'fields':['states','carbonMonoxide','voc_sum','hydrogen','water']},open('exposure_extraction_15423482_first.json','w'),indent=2)
