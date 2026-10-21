"""Frozen diagnostic: high/low CO response overlap under mixture interference.

Unit is a complete independent gas-mixture exposure. On two already seen
development datasets, use each target device's early exposures as a labeled
reference and later exposures as test. A high test exposure is called locally
ambiguous if its closest low-CO reference response is closer than its closest
high-CO reference response. Low means CO <=500 ppb, high means >=1500 ppb;
middle concentrations are excluded from the reference. Signals are signed-log
transformed and standardized from the reference, then Euclidean distance is
used. This descriptive statistic is NOT an identifiability proof, classifier,
algorithm result, or independent confirmation. It cannot determine overlap
below instrument noise without response replicate measurements.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

root = Path(__file__).parent
result = {"protocol": __doc__, "datasets": {}}
for ds, f, units in [
    ("6821340", "exposures_6821340.npz", ("A", "B", "C")),
    ("15423482", "exposures_15423482_first.npz", (
        "54118801", "54120037", "54120457", "54156953",
        "84746449", "84748697", "84749569")),
]:
    data = np.load(root / f)
    by_unit = {}
    for unit in units:
        if ds == "6821340":
            a, ya = data[f"{unit}_train_x"], data[f"{unit}_train_y"][:, 1]
            b, yb = data[f"{unit}_test_x"], data[f"{unit}_test_y"][:, 1]
        else:
            x, y = data[f"{unit}_x"], data[f"{unit}_y"]
            a, ya = x[y[:, 0] < 160], y[y[:, 0] < 160, 1]
            b, yb = x[y[:, 0] >= 160], y[y[:, 0] >= 160, 1]
        transform = lambda z: np.sign(z) * np.log1p(abs(z))
        scaler = StandardScaler().fit(transform(a))
        za, zb = scaler.transform(transform(a)), scaler.transform(transform(b))
        low, high, high_test = ya <= 500, ya >= 1500, yb >= 1500
        assert low.sum() > 0 and high.sum() > 0 and high_test.sum() > 0
        dl = pairwise_distances(zb[high_test], za[low]).min(axis=1)
        dh = pairwise_distances(zb[high_test], za[high]).min(axis=1)
        by_unit[unit] = {
            "n_low_ref": int(low.sum()), "n_high_ref": int(high.sum()),
            "n_high_test": int(high_test.sum()),
            "fraction_high_closer_to_low": round(float(np.mean(dl < dh)), 3),
            "median_low_to_high_distance_ratio": round(float(np.median(dl / dh)), 3),
        }
        print(ds, unit, by_unit[unit], flush=True)
    result["datasets"][ds] = by_unit
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2))
