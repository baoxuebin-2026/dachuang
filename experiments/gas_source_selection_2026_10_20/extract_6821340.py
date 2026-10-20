"""Exposure-level features from synchronized 3-sensor MOS mixtures.

The original train/test split is preserved; each unique `range` is one
independent mixture exposure, even though it has several temperature cycles
and is recorded simultaneously by three sensors.
"""
import h5py,json,numpy as np
SRC='zenodo_6821340_fullData.mat';ROWS=np.linspace(60,1380,16,dtype=int)
FIELDS=['range','carbon_monoxide','hydrogen','water','VOCsum']
out={}
with h5py.File(SRC) as h:
 for partition in ['train','test']:
  t=h[f'targets_{partition}'];info={k:t[k][0] for k in FIELDS};ranges=info['range']
  groups={int(r):np.flatnonzero(ranges==r) for r in np.unique(ranges[np.isfinite(ranges)])}
  for sensor in ['A','B','C']:
   refs=h[f'sensor{sensor}_{partition}'];phase=[]
   for p in range(4):
    mat=h[refs[p,0]];phase.append(np.asarray(mat[ROWS,:].T,dtype=np.float64))
   features=np.concatenate(phase,axis=1)
   xx=[];yy=[];cy=[]
   for key,ids in sorted(groups.items()):
    x=np.nanmedian(features[ids],axis=0)
    y=np.array([info[f][ids[0]] for f in FIELDS],dtype=float)
    if len(ids)>=2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
     xx.append(x);yy.append(y);cy.append(len(ids))
   stem=f'{sensor}_{partition}';out[stem+'_x']=np.array(xx);out[stem+'_y']=np.array(yy);out[stem+'_cycles']=np.array(cy)
   print(stem,len(yy),'strata',[(a,b,int(sum((out[stem+'_y'][:,0]>=a)&(out[stem+'_y'][:,0]<=b)))) for a,b in [(1,200),(201,500),(501,900)]],flush=True)
np.savez_compressed('exposures_6821340.npz',**out)
json.dump({'source':'https://doi.org/10.5281/zenodo.6821340','md5':'6ce3fefe1852219e1a66b3fcc24db045','features':'16 phase points x 4 pixels; median all stable cycles per exposure','fields':FIELDS,'rows':ROWS.tolist(),'counts':{key:int(v.shape[0]) for key,v in out.items() if key.endswith('_x')}},open('exposure_extraction_6821340.json','w'),indent=2)
