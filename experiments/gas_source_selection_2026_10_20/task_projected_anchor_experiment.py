"""Pilot test of task-projected calibration on gas-mixture exposures.

Fixed model: source batches 3+4 Ridge(alpha=10). The first 20 exposures of
each subsequent batch are calibration references with known CO, and later
exposures are scored. Unit of evaluation is exposure; batches are domains.

This is an exploratory study: all target batches have been seen before.
"""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

d=np.load('exposures_4264224.npz');result={}
for sensor in ('as_mlv','as_mlv_p2'):
    x=d[sensor+'_x'];meta=d[sensor+'_y'];batches=meta[:,0].astype(int);co=meta[:,2]
    train=np.isin(batches,[3,4]);scaler=StandardScaler().fit(x[train])
    z=scaler.transform(x);model=Ridge(alpha=10).fit(z[train],co[train]);mu=z[train].mean(axis=0)
    sensor_results={}
    for batch in (5,6,7,8,10,11,12):
        ind=np.flatnonzero(batches==batch);pilot=ind[:20];test=ind[20:]
        p_pilot=model.predict(z[pilot]);p_test=model.predict(z[test]);base_error=co[pilot]-p_pilot
        bias=float(np.mean(base_error));se=float(np.std(base_error,ddof=1)/np.sqrt(len(pilot)))
        # Predefined conservative t-like gate: avoid a correction that is within
        # twice the estimated pilot noise; never use test labels in this decision.
        corrected=p_test+bias
        gated=p_test+(bias if abs(bias)>2*se else 0.0)
        robust=p_test+np.median(base_error)
        center=model.predict(z[test]-(z[pilot].mean(axis=0)-mu))
        supervised_residual=Ridge(alpha=10).fit(z[pilot],base_error).predict(z[test])+p_test
        supervised_only=Ridge(alpha=10).fit(z[pilot],co[pilot]).predict(z[test])
        sensor_results[str(batch)]={
            'n_test':len(test),'pilot_bias_ppb':round(bias,2),'pilot_se_ppb':round(se,2),
            'gate_activated':bool(abs(bias)>2*se),
            'raw_mae':round(float(mean_absolute_error(co[test],p_test)),2),
            'unlabeled_center_mae':round(float(mean_absolute_error(co[test],center)),2),
            'pilot_mean_bias_mae':round(float(mean_absolute_error(co[test],corrected)),2),
            'pilot_median_bias_mae':round(float(mean_absolute_error(co[test],robust)),2),
            'gated_bias_mae':round(float(mean_absolute_error(co[test],gated)),2),
            'residual_ridge20_mae':round(float(mean_absolute_error(co[test],supervised_residual)),2),
            'target_ridge20_mae':round(float(mean_absolute_error(co[test],supervised_only)),2),
        }
    result[sensor]=sensor_results
print(json.dumps(result,ensure_ascii=False,indent=2))
with open('task_projected_anchor_experiment_results.json','w') as f:json.dump(result,f,ensure_ascii=False,indent=2)
