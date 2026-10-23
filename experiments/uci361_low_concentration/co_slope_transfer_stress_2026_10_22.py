"""Frozen exploratory stress: high-CO mixed-gas transfer to unseen low-CO device.

The two archives have guided prior work; this is NOT independent blind validation.
Unit: one gas exposure per target physical device. Simultaneous source-device
readings of a shared recipe are not independent exposures. No zero-CO recipe
exists, so this experiment is CO concentration regression, not gas identity.

Before looking at output: fit on source CO >=1200 ppb; target calibration is
20 medium (800 <= CO <1200) recipes sampled with seed 21; test target CO <800.
6821340 uses source official TRAIN only, pilots target TRAIN only and test
target official TEST. 15423482 uses the first calibration, split by recipe CO.
Features signed log, standardized by source train data only. Synthetic low
responses extrapolate each source device's CO response slope estimated by
Ridge(10) against CO, H2, humidity, VOC under high-CO conditions, creating
three CO levels at 25/50/75 percent of each high-CO source recipe; nuisances
remain fixed. Predictor Ridge(10), then target 20-label Ridge(10) residual.

Baselines: raw high-only source Ridge(10) with the same residual calibration,
raw high-only source HGB with the same residual calibration, target direct
Ridge(10), and constant 450 ppb (midpoint of the known 100-800 ppb test band).
All predictions clipped to [0, 2000]. Primary success requires synthetic
method mean device MAE <= 90 percent of best baseline in EACH archive, every
target device <= 110 percent of its own best baseline, and no worse mean
underprediction bias than the best baseline in either archive.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1] / "gas_source_selection_2026_10_20"
DATASETS = {
    "6821340": ["A", "B", "C"],
    "15423482_first": [
        "54118801", "54120037", "54120457", "54156953",
        "84746449", "84748697", "84749569",
    ],
}


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def rows(data, name, dataset, part):
    suffix = "_" + part if dataset == "6821340" else ""
    return slog(data[name + suffix + "_x"]), data[name + suffix + "_y"]


def run_archive(dataset, names):
    data = np.load(ROOT / f"exposures_{dataset}.npz")
    out = {}
    for target in names:
        source_x, source_y, source_id = [], [], []
        for source in names:
            if source == target:
                continue
            x, y = rows(data, source, dataset, "train")
            take = y[:, 1] >= 1200
            source_x.append(x[take]); source_y.append(y[take]); source_id.append(source)
        sx = np.vstack(source_x); sy = np.vstack(source_y)
        tx_p, ty_p = rows(data, target, dataset, "train")
        tx_t, ty_t = rows(data, target, dataset, "test")
        medium = np.where((ty_p[:, 1] >= 800) & (ty_p[:, 1] < 1200))[0]
        rng = np.random.default_rng(21)
        pilot_idx = np.sort(rng.choice(medium, size=20, replace=False))
        test_idx = np.where(ty_t[:, 1] < 800)[0]
        px, py = tx_p[pilot_idx], ty_p[pilot_idx, 1]
        test_x, test_y = tx_t[test_idx], ty_t[test_idx, 1]
        scaler = StandardScaler().fit(sx)
        sxz, pxz, txz = scaler.transform(sx), scaler.transform(px), scaler.transform(test_x)

        all_synthetic = []
        for x, y in zip(source_x, source_y):
            # Conditional CO response in ppm; nuisance columns are fitted jointly.
            z = np.column_stack((y[:, 1:4], y[:, 4])) / 1000
            xs = scaler.transform(x)
            slope = Ridge(alpha=10).fit(z, xs).coef_[:, 0]
            for factor in (0.25, 0.5, 0.75):
                level_drop = (factor - 1.0) * y[:, 1] / 1000
                all_synthetic.append((xs + level_drop[:, None] * slope[None, :],
                                      y[:, 1] * factor))
        ax = np.vstack((sxz, *(a for a, _ in all_synthetic)))
        ay = np.concatenate((sy[:, 1], *(a for _, a in all_synthetic)))

        models = {
            "slope_synthesis": Ridge(alpha=10).fit(ax, ay),
            "raw_ridge": Ridge(alpha=10).fit(sxz, sy[:, 1]),
            "raw_hgb": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                                                       min_samples_leaf=10, l2_regularization=10,
                                                       random_state=21).fit(sxz, sy[:, 1]),
        }
        preds = {"target20_ridge": Ridge(alpha=10).fit(pxz, py).predict(txz),
                 "constant_midpoint": np.full(len(test_y), 450.0)}
        for method, model in models.items():
            pilot_base = model.predict(pxz)
            residual = Ridge(alpha=10).fit(pxz, py - pilot_base)
            preds[method] = model.predict(txz) + residual.predict(txz)
        scores = {}
        for method, pred in preds.items():
            pred = np.clip(pred, 0, 2000)
            scores[method] = {
                "mae": float(mean_absolute_error(test_y, pred)),
                "underprediction": float(np.maximum(test_y - pred, 0).mean()),
            }
        out[target] = {"n_high_source": len(sy), "n_pilot": len(pilot_idx),
                       "n_low_test": len(test_idx), "scores": scores}
    methods = list(next(iter(out.values()))["scores"])
    means = {m: {k: float(np.mean([v["scores"][m][k] for v in out.values()]))
                 for k in ("mae", "underprediction")} for m in methods}
    baseline = min((m for m in methods if m != "slope_synthesis"),
                   key=lambda m: means[m]["mae"])
    device_pass = all(v["scores"]["slope_synthesis"]["mae"] <= 1.1 *
                      min(s["mae"] for m, s in v["scores"].items() if m != "slope_synthesis")
                      for v in out.values())
    success = (means["slope_synthesis"]["mae"] <= .9 * means[baseline]["mae"]
               and device_pass
               and means["slope_synthesis"]["underprediction"] <= means[baseline]["underprediction"])
    return {"devices": out, "means": means, "best_baseline": baseline,
            "device_pass": device_pass, "archive_pass": success}


if __name__ == "__main__":
    result = {name: run_archive(name, devices) for name, devices in DATASETS.items()}
    result["overall_pass"] = all(v["archive_pass"] for v in result.values())
    dest = Path(__file__).with_suffix(".json")
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: {"means": v["means"], "best_baseline": v["best_baseline"],
                           "archive_pass": v["archive_pass"]}
                      for k, v in result.items() if k != "overall_pass"}, indent=2))
    print("overall_pass", result["overall_pass"])
