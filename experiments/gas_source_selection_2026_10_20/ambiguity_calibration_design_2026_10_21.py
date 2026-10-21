"""Frozen exploratory test of CO-confounding-pair pilot recipe selection.

Unit: one complete gas-mixture exposure. Budget: 20 target-device labeled
recipes; source response and all intended gas concentrations are known before
selecting recipes. Select 8 distinct high/low CO recipe pairs (CO>=1500,
CO<=500 ppb), using source-signal proximity to expose interference. The first
pair is closest; subsequent pairs are the most separated from earlier pairs'
midpoints among the closest quartile of still-eligible pairs. Fill the remaining
four slots by source-signal farthest-first coverage. No target response or
target CO result is consulted to design the 20 experiments.

Comparators: random VOC-stratified selection (10+10), random high/low balanced
selection (8+8 and 4 middle), and all-feature farthest-first selection. Each
uses the same Ridge(10) source model and 20-label Ridge(10) target residual;
direct 20-label Ridge(10) is also evaluated. Random baselines use fixed seeds
21,37,59; mean over these three repeats is only sensitivity, not independent
replication. Test periods are held out by exposure/state/day/batch. Primary
metrics are exposure-weighted CO MAE and mean positive underestimation at
CO>=1500. Exploratory success: on each of four already seen datasets design
reduces both primary metrics >=10% compared to the best random design, with
no target device worsening >10% on MAE. These datasets have been used for
earlier hypotheses, so any success would require independent confirmation.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

P = Path(__file__).parent
log = lambda a: np.sign(a) * np.log1p(abs(a))
out = {"protocol": __doc__, "datasets": {}}

def load_cases():
    cases = []
    d = np.load(P / "exposures_6821340.npz")
    cases.append(("6821340", d["A_train_x"], d["A_train_y"],
                  [(u, d[f"{u}_train_x"], d[f"{u}_train_y"],
                    d[f"{u}_test_x"], d[f"{u}_test_y"]) for u in ("B", "C")],
                  np.arange(500), np.arange(500, 700), 1, 4))
    d = np.load(P / "exposures_15423482_first.npz")
    src = "54118801"
    sx, sy = d[f"{src}_x"], d[f"{src}_y"]
    targets = []
    for u in ("54120037", "54120457", "54156953", "84746449", "84748697", "84749569"):
        x, y = d[f"{u}_x"], d[f"{u}_y"]
        test = y[:, 0] >= 160
        targets.append((u, x, y, x[test], y[test]))
    cases.append(("15423482", sx, sy, targets,
                  np.flatnonzero(sy[:, 0] <= 100),
                  np.flatnonzero((sy[:, 0] > 100) & (sy[:, 0] < 160)), 1, 2))
    d = np.load(P / "exposures_4593853.npz")
    sy, ty = d["sensor0_y"], d["sensor1_y"]
    test = ty[:, 1] == 6
    cases.append(("4593853", d["sensor0_x"], sy,
                  [("sensor1", d["sensor1_x"], ty,
                    d["sensor1_x"][test], ty[test])],
                  np.flatnonzero(sy[:, 1] <= 4), np.flatnonzero(sy[:, 1] == 5), 2, 3))
    d = np.load(P / "exposures_4264224.npz")
    ax, ay = d["as_mlv_x"], d["as_mlv_y"]
    bx, by = d["as_mlv_p2_x"], d["as_mlv_p2_y"]
    key = lambda y: (int(y[0]), *tuple(np.round(y[2:], 3)))
    a, b = {}, {}
    for i, y in enumerate(ay): a.setdefault(key(y), []).append(i)
    for i, y in enumerate(by): b.setdefault(key(y), []).append(i)
    keys = sorted(k for k in a.keys() & b.keys() if len(a[k]) == len(b[k]) == 1)
    ii, jj = [a[k][0] for k in keys], [b[k][0] for k in keys]
    sx, sy, tx, ty = ax[ii], ay[ii], bx[jj], by[jj]
    test = ty[:, 0] >= 6
    cases.append(("4264224", sx, sy,
                  [("as_mlv_p2", tx, ty, tx[test], ty[test])],
                  np.flatnonzero(sy[:, 0] <= 4), np.flatnonzero(sy[:, 0] == 5), 2, 5))
    return cases

def farthest(z, selected, count):
    selected = list(selected)
    if not selected:
        selected = [int(np.argmax(np.linalg.norm(z - np.mean(z, axis=0), axis=1)))]
    while len(selected) < count:
        mind = pairwise_distances(z, z[selected]).min(axis=1)
        mind[selected] = -1
        selected.append(int(np.argmax(mind)))
    return np.array(selected)

def choose(z, y):
    low, high = np.flatnonzero(y <= 500), np.flatnonzero(y >= 1500)
    dist = pairwise_distances(z[low], z[high])
    used_low, used_high, chosen, middles = set(), set(), [], []
    for turn in range(8):
        options = [(float(dist[i, j]), int(l), int(h))
                   for i, l in enumerate(low) for j, h in enumerate(high)
                   if l not in used_low and h not in used_high]
        options.sort()
        if turn == 0:
            _, l, h = options[0]
        else:
            nearest_quartile = options[:max(1, int(np.ceil(len(options) / 4)))]
            def pair_diversity(pair):
                _, lo, hi = pair
                midpoint = (z[lo] + z[hi]) / 2
                return float(pairwise_distances(midpoint[None], np.array(middles)).min())
            _, l, h = max(nearest_quartile, key=pair_diversity)
        chosen.extend((l, h)); used_low.add(l); used_high.add(h)
        middles.append((z[l] + z[h]) / 2)
    return farthest(z, chosen, 20)

def score(y, pred):
    p = np.clip(pred, 0, 2000)
    high, low = y >= 1500, y <= 500
    return {"mae": float(np.mean(abs(y - p))),
            "high_under": float(np.mean(np.maximum(0, y[high] - p[high]))),
            "high_miss": float(np.mean(p[high] < 1000)),
            "low_false_alarm": float(np.mean(p[low] >= 1000))}

for name, sx, sy, targets, tr, pool, co, voc in load_cases():
    source_scale = StandardScaler().fit(log(sx[tr]))
    source = Ridge(alpha=10).fit(source_scale.transform(log(sx[tr])), sy[tr, co])
    z = source_scale.transform(log(sx[pool]))
    y, v = sy[pool, co], sy[pool, voc]
    low, high, middle = [np.flatnonzero(mask) for mask in
                         (y <= 500, y >= 1500, (y > 500) & (y < 1500))]
    assert min(len(low), len(high)) >= 8 and len(middle) >= 4
    choices = {"confounding_pairs": [choose(z, y)],
               "farthest_first": [farthest(z, [], 20)],
               "random_high_low": [], "random_voc": []}
    vl, vh = np.flatnonzero(v <= np.median(v)), np.flatnonzero(v > np.median(v))
    for seed in (21, 37, 59):
        rng = np.random.default_rng(seed)
        choices["random_high_low"].append(np.r_[rng.choice(low, 8, False),
                                                 rng.choice(high, 8, False),
                                                 rng.choice(middle, 4, False)])
        choices["random_voc"].append(np.r_[rng.choice(vl, 10, False),
                                           rng.choice(vh, 10, False)])
    results = {}
    for u, tx, ty, test_x, test_y in targets:
        assert np.allclose(sy[:, co], ty[:, co])
        test_z = source_scale.transform(log(test_x))
        base_pred = source.predict(test_z)
        unit = {}
        for method, arr in choices.items():
            runs = []
            for selected in arr:
                pilot = pool[selected]
                target_pilot_z = source_scale.transform(log(tx[pilot]))
                pilot_y = ty[pilot, co]
                residual = Ridge(alpha=10).fit(target_pilot_z,
                          pilot_y - source.predict(target_pilot_z))
                p_resid = base_pred + residual.predict(test_z)
                direct_scale = StandardScaler().fit(log(tx[pilot]))
                direct = Ridge(alpha=10).fit(direct_scale.transform(log(tx[pilot])),
                                             pilot_y)
                p_direct = direct.predict(direct_scale.transform(log(test_x)))
                runs.append({"residual": score(test_y[:, co], p_resid),
                             "direct": score(test_y[:, co], p_direct)})
            unit[method] = {model: {metric: round(float(np.mean([
                run[model][metric] for run in runs])), 3)
                for metric in runs[0][model]} for model in ("residual", "direct")}
        results[u] = unit
    means = {m: {model: {metric: round(float(np.mean([
        unit[m][model][metric] for unit in results.values()])), 3)
        for metric in ("mae", "high_under", "high_miss", "low_false_alarm")}
        for model in ("residual", "direct")} for m in choices}
    base = min(("random_voc", "random_high_low"),
               key=lambda m: means[m]["residual"]["mae"])
    win = (means["confounding_pairs"]["residual"]["mae"] <=
           0.9 * means[base]["residual"]["mae"] and
           means["confounding_pairs"]["residual"]["high_under"] <=
           0.9 * means[base]["residual"]["high_under"] and
           all(unit["confounding_pairs"]["residual"]["mae"] <=
               1.1 * unit[base]["residual"]["mae"] for unit in results.values()))
    out["datasets"][name] = {"reference_n": len(tr), "pilot_pool_n": len(pool),
                              "selected_recipes": pool[choices["confounding_pairs"][0]].tolist(),
                              "per_target": results, "mean": means,
                              "best_random": base, "pass": bool(win)}
    print(name, "n", len(tr), len(pool), "best", base,
          "scores", {k: v["residual"] for k, v in means.items()}, "PASS", win, flush=True)
out["pass"] = all(d["pass"] for d in out["datasets"].values())
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2))
print("GLOBAL PASS", out["pass"], flush=True)
