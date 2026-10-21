"""FROZEN external decision test of paired-shift gate after 682/154 ideation.

Device offset ratio = RMS over features of median paired source-target
standardized signed-log response displacement / RMS of source response
variation over eligible pilot pool. Rule before reading these outcomes:
if ratio > 1.5 choose GLOBAL paired median response alignment; otherwise
choose 20-CO-label residual Ridge(alpha10). Source Ridge(alpha10) trained
on source-only earlier lab/batches. 20 target pilot recipes stratified
10/10 by TVOC, seeds 21/37/59, each test set scored once per seed.

Dataset 4593853: sensor0 source labs1-4, matched pilot lab5, target sensor1
lab6 test. Dataset 4264224: source as_mlv batches1-4, matched pilot batch5,
target as_mlv_p2 batches6-12 test. Match independent exposures by batch and
gas composition CO/H2/water/TVOC (unique keys only). These data sources had
other analyses but not this pairing/shrinkage rule. Source device is never
trained on target test labels. Benchmark always-residual, always-map, target
direct20. Success requires gated mean MAE >=10% below always-residual across
both datasets combined, neither dataset worsens >10%, high CO>=1500 positive
underestimation does not worsen pooled, and at least one dataset chooses map.
3 pilot seeds are repeat sensitivity, NOT three independent validations.
Do not adjust threshold after results.
"""
from pathlib import Path
import numpy as np,json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

R=Path(__file__).parent;out={'protocol':__doc__,'datasets':{}}
log=lambda x:np.sign(x)*np.log1p(abs(x))
def metric(y,p):
 p=np.clip(p,0,2000);high=y>=1500
 return {'mae':round(float(abs(y-p).mean()),2),'high_under':round(float(np.maximum(0,y[high]-p[high]).mean()),2)}

# Build two independent, correctly paired data units.
data=np.load(R/'exposures_4593853.npz')
sx=data['sensor0_x'];sy=data['sensor0_y'];tx=data['sensor1_x'];ty=data['sensor1_y']
assert len(sy)==len(ty) and np.allclose(sy[:,2:],ty[:,2:])
cases=[{'name':'4593853','sx':sx,'sy':sy,'tx':tx,'ty':ty,
 'train':np.where(sy[:,1]<=4)[0],'pool':np.where(sy[:,1]==5)[0],
 'test':np.where(sy[:,1]==6)[0],'co_col':2,'voc_col':3}]

data=np.load(R/'exposures_4264224.npz')
ax=data['as_mlv_x'];ay=data['as_mlv_y'];bx=data['as_mlv_p2_x'];by=data['as_mlv_p2_y']
def key(y):return (int(y[0]),*tuple(np.round(y[2:],3)))
ai={};bi={}
for i,row in enumerate(ay):ai.setdefault(key(row),[]).append(i)
for i,row in enumerate(by):bi.setdefault(key(row),[]).append(i)
keys=sorted(k for k in ai.keys()&bi.keys() if len(ai[k])==1 and len(bi[k])==1)
assert len(keys)>1000,len(keys)
ii=[ai[k][0] for k in keys];jj=[bi[k][0] for k in keys]
sx=ax[ii];sy=ay[ii];tx=bx[jj];ty=by[jj]
assert np.allclose(sy[:,2:],ty[:,2:])
cases.append({'name':'4264224','sx':sx,'sy':sy,'tx':tx,'ty':ty,
 'train':np.where(sy[:,0]<=4)[0],'pool':np.where(sy[:,0]==5)[0],
 'test':np.where(sy[:,0]>=6)[0],'co_col':2,'voc_col':5})

for case in cases:
 name=case['name'];sx=case['sx'];tx=case['tx'];sy=case['sy'];ty=case['ty']
 train=case['train'];pool=case['pool'];te=case['test'];co=case['co_col'];voc=case['voc_col']
 scale=StandardScaler().fit(log(sx[train]));src=scale.transform(log(sx[train]))
 model=Ridge(alpha=10).fit(src,sy[train,co])
 src_pool=scale.transform(log(sx[pool]));target_pool=scale.transform(log(tx[pool]))
 displacement=src_pool-target_pool
 offset=np.sqrt(np.mean(np.median(displacement,axis=0)**2))
 variability=np.sqrt(np.mean(np.var(src_pool,axis=0)))
 shift_ratio=offset/variability
 v=ty[pool,voc];low=pool[v<=np.median(v)];high=pool[v>np.median(v)]
 runs=[]
 for seed in (21,37,59):
  rng=np.random.default_rng(seed);p=np.r_[rng.choice(low,10,False),rng.choice(high,10,False)]
  z=log(tx[p]);s=log(sx[p]);test=log(tx[te]);y=ty[p,co];truth=ty[te,co]
  zp=scale.transform(z);zt=scale.transform(test)
  d=np.median(s-z,axis=0)
  aligned=model.predict(scale.transform(test+d))
  corrected=model.predict(zt)+Ridge(alpha=10).fit(zp,y-model.predict(zp)).predict(zt)
  target_sc=StandardScaler().fit(z)
  direct=Ridge(alpha=10).fit(target_sc.transform(z),y).predict(target_sc.transform(test))
  decision='paired_map' if shift_ratio>1.5 else 'residual'
  scores={'paired_map':metric(truth,aligned),'residual':metric(truth,corrected),'direct20':metric(truth,direct)}
  runs.append({'seed':seed,'decision':decision,'scores':scores})
 stats={k:{field:round(float(np.mean([r['scores'][k][field] for r in runs])),2) for field in ('mae','high_under')} for k in ('paired_map','residual','direct20')}
 out['datasets'][name]={'n_source_train':len(train),'n_pilot_pool':len(pool),'n_test':len(te),
  'shift_ratio':round(float(shift_ratio),3),'decision':runs[0]['decision'],'metrics':stats,'runs':runs}
 print(name,{k:v for k,v in out['datasets'][name].items() if k!='runs'},flush=True)

baseline=np.mean([d['metrics']['residual']['mae'] for d in out['datasets'].values()])
selected=np.mean([d['metrics'][d['decision']]['mae'] for d in out['datasets'].values()])
under_base=np.mean([d['metrics']['residual']['high_under'] for d in out['datasets'].values()])
under_sel=np.mean([d['metrics'][d['decision']]['high_under'] for d in out['datasets'].values()])
out['pass']=bool(selected<=.9*baseline and under_sel<=under_base and any(d['decision']=='paired_map' for d in out['datasets'].values()) and all(d['metrics'][d['decision']]['mae']<=1.1*d['metrics']['residual']['mae'] for d in out['datasets'].values()))
out['pooled']={'baseline_mae':round(float(baseline),2),'selected_mae':round(float(selected),2),
 'baseline_high_under':round(float(under_base),2),'selected_high_under':round(float(under_sel),2)}
print('POOLED',out['pooled'],'PASS',out['pass'],flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
