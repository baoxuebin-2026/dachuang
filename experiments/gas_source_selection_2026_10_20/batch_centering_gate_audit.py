"""Exploratory gate diagnostics; batch 6/7/8 were previously inspected.

The output is NOT a blind validation of any rule chosen from these results.
"""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

data=np.load('exposures_4264224.npz')
out={}
for sensor in ['as_mlv','as_mlv_p2']:
    x=data[sensor+'_x'];labels=data[sensor+'_y']
    batch=labels[:,0].astype(int);y=labels[:,2]
    good=(batch==3)|(batch==4)
    scaler=StandardScaler().fit(x[good]);z=scaler.transform(x)
    model=Ridge(alpha=10).fit(z[good],y[good])
    src=z[good];mu=src.mean(0);cov=np.cov(src,rowvar=False)
    sensor_result={}
    for b in [5,6,7,8,9,10,11,12]:
        which=np.flatnonzero(batch==b)
        if len(which)<50:continue
        warm=which[:20];late=which[20:]
        zz=z[which];shift=z[warm].mean(0)-mu
        raw=model.predict(z[late]);corrected=model.predict(z[late]-shift)
        cv=np.cov(zz,rowvar=False)
        cov_ratio=float(np.linalg.norm(cv-cov)/np.linalg.norm(cov))
        # Confidence proxies measured without target labels:
        # ratio of target centered to source covariance; warmup split stability.
        split=float(abs(model.predict(z[warm[:10]]).mean()-model.predict(z[warm[10:]]).mean()))
        target_spread=float(np.std(model.predict(zz)))
        src_spread=float(np.std(model.predict(src)))
        sensor_result[b]={
            'n_warm':len(warm),'n_test':len(late),
            'raw_mae':round(float(mean_absolute_error(y[late],raw)),3),
            'centered_mae':round(float(mean_absolute_error(y[late],corrected)),3),
            'delta':round(float(mean_absolute_error(y[late],corrected)-mean_absolute_error(y[late],raw)),3),
            'cov_ratio':round(cov_ratio,3),'split_prediction_difference':round(split,3),
            'target_to_source_pred_spread':round(target_spread/src_spread,3),
            # Diagnostic only: target gas distribution is not known at inference.
            'warm_co_mean':round(float(y[warm].mean()),2),
            'late_co_mean':round(float(y[late].mean()),2),
        }
    out[sensor]=sensor_result
print(json.dumps(out,indent=2))
with open('batch_centering_gate_audit_results.json','w') as f:json.dump(out,f,indent=2)
