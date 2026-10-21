"""Protocol frozen before examining UCI270 response features or test labels.

UCI270 has six pure gases (no CO), one array measured over 36 months, and
precomputed 8 features per sensor. This is a *mechanism stress test* of
response scaling across time and concentration, not cross-device CO proof.
Units are complete measurements, never feature rows/windows. Source batches
1..5, target batches 6..10. For each class, choose training concentrations
at or above the source-only 2/3 quantile; test at or below the source-only
1/3 quantile. Concentrations select protocol only, never input features.
Drop every sensor's already normalized second feature; remaining 7 response
features per sensor are linearly scaled (0.5,0.75,1.0) at training only.
Apply fixed signed log1p before train-fitted StandardScaler, SVC RBF C=1.
Compare same SVC with no augmentation and a row-L2-normalized SVC. No
hyperparameter or threshold choice based on target features/outcomes.

Failure: augmented SVC does not beat best fixed baseline by 0.03 macro F1,
or loses >0.05 recall in any gas, or improves fewer than 4/5 target batches.
Batch test sets overlap in condition, so batches are the evaluation blocks.
"""

import json
import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get("UCI270_ARCHIVE", ROOT / "uci270.zip"))
EXPECTED_SHA256 = "98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e"


def load():
    xs, ys, concentrations, batches = [], [], [], []
    with zipfile.ZipFile(ARCHIVE) as z:
        for batch in range(1, 11):
            for line in z.open(f"batch{batch}.dat"):
                label, *features = line.decode().split()
                gas, conc = label.split(";")
                vector = np.zeros(128)
                for item in features:
                    index, value = item.split(":")
                    vector[int(index) - 1] = float(value)
                xs.append(vector.reshape(16, 8)[:, [0, 2, 3, 4, 5, 6, 7]].ravel())
                ys.append(int(gas) - 1)
                concentrations.append(float(conc))
                batches.append(batch)
    return np.asarray(xs), np.asarray(ys), np.asarray(concentrations), np.asarray(batches)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def report(y, p, b):
    return {
        "macro_f1": float(f1_score(y, p, labels=range(6), average="macro", zero_division=0)),
        "class_recall": recall_score(y, p, labels=range(6), average=None, zero_division=0).tolist(),
        "by_batch_f1": {str(k): float(f1_score(y[b == k], p[b == k], labels=range(6), average="macro", zero_division=0)) for k in range(6, 11)},
        "confusion": confusion_matrix(y, p, labels=range(6)).tolist(),
    }


def main():
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected archive: {digest}")
    x, y, c, b = load()
    low, high = [], []
    for gas in range(6):
        source = c[(b <= 5) & (y == gas)]
        low.append(float(np.quantile(source, 1 / 3)))
        high.append(float(np.quantile(source, 2 / 3)))
    tr = (b <= 5) & (c >= np.asarray(high)[y])
    te = (b >= 6) & (c <= np.asarray(low)[y])
    if min(np.bincount(y[tr], minlength=6)) < 15 or min(np.bincount(y[te], minlength=6)) < 15:
        raise ValueError("One of six gases has fewer than 15 complete source/test exposures")
    source, target = signed_log(x[tr]), signed_log(x[te])
    # Augmentation precedes signed-log; low-power MOX response features scale
    # in the original feature domain, while the transform tames outliers.
    aug = signed_log(np.concatenate([x[tr] * f for f in (0.5, 0.75, 1.0)]))
    labels = np.tile(y[tr], 3)
    models = {
        "unaugmented_svm": (make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")), source, y[tr], target),
        "amplitude_augmented_svm": (make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")), aug, labels, target),
        "unit_norm_svm": (make_pipeline(Normalizer(), StandardScaler(), SVC(C=1, gamma="scale")), source, y[tr], target),
    }
    results = {}
    for name, (model, features, yy, unseen) in models.items():
        model.fit(features, yy)
        results[name] = report(y[te], model.predict(unseen), b[te])
        print(name, results[name]["macro_f1"], flush=True)
    best = max((results[k] for k in ("unaugmented_svm", "unit_norm_svm")), key=lambda z: z["macro_f1"])
    ours = results["amplitude_augmented_svm"]
    success = (
        ours["macro_f1"] > best["macro_f1"] + 0.03
        and all(a >= q - 0.05 for a, q in zip(ours["class_recall"], best["class_recall"]))
        and sum(ours["by_batch_f1"][str(k)] > best["by_batch_f1"][str(k)] for k in range(6, 11)) >= 4
    )
    output = {"protocol": __doc__, "archive_sha256": digest, "n_train": int(tr.sum()), "n_test": int(te.sum()), "train_per_gas": np.bincount(y[tr], minlength=6).tolist(), "test_per_gas": np.bincount(y[te], minlength=6).tolist(), "low": low, "high": high, "passes": success, "results": results}
    (ROOT / "uci270_amplitude_stress.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print("passes", success, "counts", output["train_per_gas"], output["test_per_gas"])


if __name__ == "__main__":
    main()
