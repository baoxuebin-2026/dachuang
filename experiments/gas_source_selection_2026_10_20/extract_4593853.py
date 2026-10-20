"""Exposure-level features from original Zenodo 4593853 HDF5 MATLAB data."""
import h5py
import numpy as np
import json

PATH='zenodo_4593853_data.mat'
ROWS=np.linspace(100,2300,16,dtype=int)
FIELDS=['range','lab','carbon_monoxide','TVOCppb','water','hydrogen']
output={}
with h5py.File(PATH) as h:
 for name in ('sensor0','sensor1'):
  g=h['data'][name];t=g['targets'];info={k:t[k][0] for k in FIELDS};n=len(info['range'])
  valid=np.isfinite(info['range'])&np.isfinite(info['lab'])&np.isfinite(info['carbon_monoxide'])
  valid&=np.all([np.isfinite(info[k]) for k in FIELDS],axis=0)
  all_phase=np.empty((n,4*len(ROWS)),dtype=np.float32);all_phase.fill(np.nan)
  for p in range(4):
   ds=g[f'resistance{p}']
   for start in range(0,n,600):
    stop=min(start+600,n)
    block=ds[:,start:stop]
    v=block[ROWS,:].T
    with np.errstate(invalid='ignore',divide='ignore'):
     all_phase[start:stop,p*len(ROWS):(p+1)*len(ROWS)]=-np.log10(np.maximum(v,1e-12))
   print(name,'pixel',p,flush=True)
  features=[];labels=[];cycles=[]
  for range_id in np.unique(info['range'][valid]):
   idx=np.flatnonzero(valid&(info['range']==range_id))
   idx=idx[len(idx)//2:]
   x=np.nanmedian(all_phase[idx],axis=0)
   y=np.array([info[k][idx[0]] for k in FIELDS],dtype=float)
   if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
    features.append(x);labels.append(y);cycles.append(len(idx))
  output[name+'_x']=np.asarray(features);output[name+'_y']=np.asarray(labels);output[name+'_cycles']=np.asarray(cycles)
  ll=output[name+'_y'][:,1]
  print(name,'exposures',len(ll),'per-lab',[(i,int(sum(ll==i))) for i in np.unique(ll)],flush=True)
np.savez_compressed('exposures_4593853.npz',**output)
json.dump({'source':'https://doi.org/10.5281/zenodo.4593853','md5':'eea034a6cc1881b0988cb1dc99c2799e','fields':FIELDS,'rows':ROWS.tolist(),'exposure_counts':{s:int(output[s+'_x'].shape[0]) for s in ['sensor0','sensor1']}},open('exposure_extraction_4593853.json','w'),indent=2)
