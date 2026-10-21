"""Frozen 14-device Ridge/ExtraTrees blend before evaluation.

Target device is wholly held out. Six remaining same-type devices are source.
Inner threefold device-group CV picks Ridge weight in {0,.25,.5,.75,1}
minimizing worst source-device MAE. Compare with fixed Ridge, ExtraTrees,
and 50/50, all using same features and model settings as earlier UCI487 audit.
Success: pooled selected MAE at least 10% lower than best fixed, at least 5%
lower in each hardware family, and worst-device MAE no worse. Shared physical
gas exposures mean 14 sensors are not 14 independent mixture datasets.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

root = Path(__file__).resolve().parents[2]
data = pd.read_csv(root / 'results/uci487_v3/segment_features.csv')
X = np.concatenate([data[[f'log_r{u}_high', f'log_r{u}_low', 'humidity_pct']].to_numpy() for u in range(1, 15)])
y = np.tile(data.co_reference_ppm.to_numpy(), 14)
units = np.repeat(np.arange(1, 15), len(data))
weights = (0., .25, .5, .75, 1.)
models = {
    'ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
    'extra_trees': lambda: ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=1., random_state=21, n_jobs=1),
}
output = {'protocol': __doc__, 'devices': {}}
for target in range(1, 15):
    family = range(1, 8) if target <= 7 else range(8, 15)
    train = np.where(np.isin(units, family) & (units != target))[0]
    test = np.where(units == target)[0]
    inner = {name: np.full(len(train), np.nan) for name in models}
    for i, j in GroupKFold(3).split(X[train], y[train], units[train]):
        for name, create in models.items():
            inner[name][j] = np.clip(create().fit(X[train[i]], y[train[i]]).predict(X[train[j]]), 0, 20)
    def worst(weight):
        prediction = weight * inner['ridge'] + (1 - weight) * inner['extra_trees']
        return max(np.abs(y[train][units[train] == u] - prediction[units[train] == u]).mean() for u in family if u != target)
    weight = min(weights, key=lambda w: (worst(w), -w))
    pred = {name: np.clip(create().fit(X[train], y[train]).predict(X[test]), 0, 20) for name, create in models.items()}
    pred['half'] = .5 * (pred['ridge'] + pred['extra_trees'])
    pred['selected'] = weight * pred['ridge'] + (1 - weight) * pred['extra_trees']
    errors = {name: round(float(np.abs(y[test] - p).mean()), 3) for name, p in pred.items()}
    output['devices'][str(target)] = {'ridge_weight': weight, 'mae': errors}
    print(target, weight, errors, flush=True)
scopes = {'all': range(1, 15), 'figaro': range(1, 8), 'fis': range(8, 15)}
output['summary'] = {
    name: {**{scope: round(float(np.mean([output['devices'][str(u)]['mae'][name] for u in ids])), 3) for scope, ids in scopes.items()},
           'worst': round(float(max(output['devices'][str(u)]['mae'][name] for u in range(1, 15))), 3)}
    for name in ('ridge', 'extra_trees', 'half', 'selected')
}
fixed = ('ridge', 'extra_trees', 'half')
best = {scope: min(output['summary'][name][scope] for name in fixed) for scope in ('all', 'figaro', 'fis', 'worst')}
selected = output['summary']['selected']
output['pass'] = selected['all'] <= .9 * best['all'] and selected['figaro'] <= .95 * best['figaro'] and selected['fis'] <= .95 * best['fis'] and selected['worst'] <= best['worst']
print('SUMMARY', output['summary'], 'BEST', best, 'PASS', output['pass'], flush=True)
(Path(__file__).with_suffix('.json')).write_text(json.dumps(output, indent=2))
