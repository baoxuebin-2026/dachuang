"""Frozen exploratory factorial forward/inverse model of CO+CH4+H2S.

Data: Mendeley 10.17632/2mpx85xt6g.1, 186 complete mixture recordings,
never window-split. From each complete recording use 3 channels' baseline,
and signed-log deltas of three disjoint exposure periods. Forward kernel
ridge maps known (CH4,CO,H2S) compositions onto standardized sensor response;
at test, among 216 candidate gas compositions, select the one with smallest
noise-scaled sensor response distance. Fixed RBF gamma=1/3, alpha=0.1.
Compare ExtraTrees(300), RBF SVR(C=100,epsilon=.1), and Ridge(10) direct CO
regression on exactly the same file features. Three fixed shuffled 5-fold
file splits seeds11/21/37. Evaluate file-level CO MAE and high>=2000 ppm
positive underestimation. Pass only if both metrics improve >=10% against
the BEST of ET/SVR in each seed; no held-out recipe labels enter fitting.
This is a new hypothesis, not an independent test or a claim of novelty.
"""
from pathlib import Path
import json
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

ps = sorted(Path(sys.argv[1]).rglob("*.txt"))
assert len(ps) == 186
features, labels = [], []
for p in ps:
    with p.open() as f:
        a = np.array([[float(v) for v in f.readline().split()] for _ in range(500)])
    base = np.median(a[:50], axis=0)
    snapshots = [np.median(a[s:s + 50], axis=0) for s in (100, 200, 300)]
    delta = [np.sign(x - base) * np.log1p(abs(x - base)) for x in snapshots]
    features.append(np.r_[np.sign(base) * np.log1p(abs(base)), *delta])
    labels.append([int(v) for v in p.stem.split("-")])
X, Y = np.array(features), np.array(labels)
assert np.isfinite(X).all() and np.isfinite(Y).all()
grid = np.array(np.meshgrid(np.arange(0, 10001, 2000),
                            np.arange(0, 2501, 500),
                            np.arange(0, 26, 5))).reshape(3, -1).T
scale_gas = np.array([10000, 2500, 25])

def metrics(y, p):
    p = np.clip(p, 0, 2500)
    return [float(np.mean(abs(y - p))),
            float(np.mean(np.maximum(0, y[y >= 2000] - p[y >= 2000])))]

output = {"protocol": __doc__, "n_files": len(ps), "results": {}}
for seed in (11, 21, 37):
    predictions = {k: np.full(len(Y), np.nan) for k in
                   ("forward_inverse", "extra_trees", "svr", "ridge")}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, Y[:, 1]):
        xs = StandardScaler().fit(X[tr])
        xtr, xte = xs.transform(X[tr]), xs.transform(X[te])
        forward = KernelRidge(kernel="rbf", gamma=1/3, alpha=.1)
        forward.fit(Y[tr] / scale_gas, xtr)
        predicted_responses = forward.predict(grid / scale_gas)
        distances = pairwise_distances(xte, predicted_responses)
        predictions["forward_inverse"][te] = grid[np.argmin(distances, axis=1), 1]
        predictions["extra_trees"][te] = ExtraTreesRegressor(
            n_estimators=300, max_features=.8, random_state=21, n_jobs=1
        ).fit(xtr, Y[tr, 1]).predict(xte)
        predictions["svr"][te] = SVR(C=100, epsilon=.1, gamma="scale").fit(
            xtr, Y[tr, 1] / 1000).predict(xte) * 1000
        predictions["ridge"][te] = Ridge(alpha=10).fit(xtr, Y[tr, 1]).predict(xte)
    row = {m: dict(zip(("mae", "high_under"),
                       [round(v, 2) for v in metrics(Y[:, 1], pred)]))
           for m, pred in predictions.items()}
    best = min(("extra_trees", "svr"), key=lambda m: row[m]["mae"])
    row["best_nonlinear"] = best
    row["pass"] = bool(all(row["forward_inverse"][metric] <=
                           .9 * row[best][metric] for metric in ("mae", "high_under")))
    output["results"][str(seed)] = row
    print(seed, row, flush=True)
output["pass"] = all(r["pass"] for r in output["results"].values())
print("GLOBAL PASS", output["pass"], flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(output, indent=2))
