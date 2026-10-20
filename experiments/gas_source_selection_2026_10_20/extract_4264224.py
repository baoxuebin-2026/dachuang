"""Reduce original Zenodo 4264224 cycles to exposure-level phase features.

Independent evaluation unit: gas-mixture exposure (range), not sensor cycle.
"""
import json
import h5py
import numpy as np

FIELDS = ['measurement','range','carbon_monoxide','hydrogen','water','tvoc_ppb']
PHASE_ROWS = np.linspace(500,11500,16,dtype=int)

with h5py.File('dataset_4264224.mat') as f:
    output={}
    for name in ['as_mlv','as_mlv_p2']:
        target=f[name+'_targets']
        meta={k:np.asarray(target[k][0]) for k in FIELDS}
        valid=np.isfinite(meta['range']) & np.isfinite(meta['carbon_monoxide'])
        valid &= np.isfinite(meta['hydrogen']) & np.isfinite(meta['tvoc_ppb']) & np.isfinite(meta['water'])
        cols=np.flatnonzero(valid)
        data=f[name]
        # HDF5 is chunked by row in AS-MLV and by cycle in AS-MLV-P2.
        if name=='as_mlv':
            phases=np.stack([data[int(row),cols] for row in PHASE_ROWS],axis=1)
        else:
            phases=np.empty((len(cols),len(PHASE_ROWS)))
            for p, col in enumerate(cols):
                phases[p]=data[PHASE_ROWS,int(col)]
                if p%2000==0:print(name,p,len(cols),flush=True)
        groups={}
        for pos,col in enumerate(cols):
            key=int(meta['range'][col])
            groups.setdefault(key,[]).append(pos)
        features=[]; labels=[]; counts=[]
        for key,positions in sorted(groups.items()):
            position=np.array(positions)
            # Exclude first half of cycles when computing response; no target labels
            # or batch-specific statistics enter this feature extraction.
            position=position[len(position)//2:]
            x=np.nanmedian(phases[position],axis=0)
            idx=cols[positions[0]]
            y=np.array([meta[k][idx] for k in FIELDS],dtype=float)
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                features.append(x);labels.append(y);counts.append(len(positions))
        output[name+'_x']=np.asarray(features)
        output[name+'_y']=np.asarray(labels)
        output[name+'_cycles']=np.asarray(counts)
        print(name, len(labels), {int(m):int(sum(np.asarray(labels)[:,0]==m)) for m in np.unique(np.asarray(labels)[:,0])},flush=True)
    np.savez_compressed('exposures_4264224.npz',**output)
    with open('exposure_extraction_4264224.json','w') as out:
        json.dump({'source':'https://doi.org/10.5281/zenodo.4264224','source_md5':'7e5e62d12a7a3f3d751634aa36e0633d','fields':FIELDS,'phase_rows':PHASE_ROWS.tolist(),'exposures':{k:int(v.shape[0]) for k,v in output.items() if k.endswith('_x')}},out,indent=2)
