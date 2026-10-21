"""Frozen multi-device CO disagreement pilot on two independent data sources.

6821340: three devices, 700 training recipes, official 200 test recipes.
15423482: seven devices, recipe state <=100 train, >=160 test, others omitted.
Fit separate ExtraTrees(200, leaf3, .8 features) per device on log signals;
CO ppm/ppb units follow original labels. Clip at [0, source train max].
Same-recipe median prediction and median absolute deviation across devices.
Top 25% disagreement is a fixed-capacity inspection queue; evaluate its
coverage of top 20% median-ensemble errors and mean-error enrichment.
Success requires >=40% top-error capture and >=1.5 mean-error enrichment in
each independent source. Compare random 25% queue (20% capture, 1x mean).
No learning/tuning from test labels. Do not count simultaneous devices as
independent gas exposures. A successful ranking would warrant further checks
against OOD and within-model uncertainty, not prove originality or safety.
"""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesRegressor

R=Path(__file__).parent
settings=[('6821340','exposures_6821340.npz',['A','B','C']),('15423482_first','exposures_15423482_first.npz',['54118801','54120037','54120457','54156953','84746449','84748697','84749569'])]
out={'protocol':__doc__,'datasets':{}}
for key,filename,devices in settings:
 d=np.load(R/filename)
 forecast=[];truth=None;unit_errors=[]
 for unit in devices:
  if key=='6821340':
   Xtr=d[f'{unit}_train_x'];Ytr=d[f'{unit}_train_y'][:,1];Xte=d[f'{unit}_test_x'];Yte=d[f'{unit}_test_y'][:,1]
  else:
   X=d[f'{unit}_x'];info=d[f'{unit}_y'];Y=info[:,1]
   tr=info[:,0]<=100;te=info[:,0]>=160
   Xtr=X[tr];Ytr=Y[tr];Xte=X[te];Yte=Y[te]
  assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
  signed_log=lambda v:np.sign(v)*np.log1p(np.abs(v))
  if truth is None:truth=Yte
  else:assert np.allclose(truth,Yte)
  model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=.8,random_state=21,n_jobs=1)
  pred=np.clip(model.fit(signed_log(Xtr),Ytr).predict(signed_log(Xte)),0,np.max(Ytr))
  forecast.append(pred);unit_errors.append(round(float(np.abs(pred-Yte).mean()),3))
 P=np.stack(forecast);median=np.median(P,axis=0);error=np.abs(median-truth)
 disagree=np.median(np.abs(P-median),axis=0)
 n=len(truth);queue=np.argsort(-disagree)[:int(np.ceil(.25*n))];hard=np.argsort(-error)[:int(np.ceil(.2*n))]
 selected=np.zeros(n,dtype=bool);selected[queue]=True
 capture=float(np.isin(hard,queue).mean());enrich=float(np.mean(error[selected])/np.mean(error[~selected]))
 result={'n_test_exposures':n,'devices':len(devices),'individual_mae':unit_errors,
   'median_mae':round(float(error.mean()),3),'mean_prediction_mae':round(float(np.abs(P.mean(0)-truth).mean()),3),
   'disagreement_top_quarter_hard_error_capture':round(capture,3),
   'disagreement_top_quarter_error_enrichment':round(enrich,3),
   'hard_error_cut':round(float(error[hard[-1]]),3),
   'co_zero_false':round(float(median[truth==0].mean()),3) if any(truth==0) else None}
 print(key,result,flush=True);out['datasets'][key]=result
out['pass']=all(v['disagreement_top_quarter_hard_error_capture']>=.4 and v['disagreement_top_quarter_error_enrichment']>=1.5 for v in out['datasets'].values())
print('PASS',out['pass'],flush=True)
(Path(__file__).with_suffix('.json')).write_text(json.dumps(out,indent=2))
