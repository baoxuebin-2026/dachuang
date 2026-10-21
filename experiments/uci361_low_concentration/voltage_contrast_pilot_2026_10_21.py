"""Frozen exploratory protocol, written before loading UCI361 responses.

Unit: complete exposure, five leave-one-device-out folds. Train only source
devices' concentration ranks 4..10, test held device ranks 1..3. Compare
identical 0.5,0.75,1 global response scaling for RBF SVM. Candidate augments
its 160-D input with each matched sensor's two-heater contrast, normalized by
the response RMS of the same trial plus 0.05. This ratio removes the shared
concentration gain and tests whether two-heater selectivity survives transfer.
No target-device sample (labeled or unlabeled) is used for preprocessing.

Failure if macro F1 <= baseline, any CO recall lost, CO false positives rise,
or fewer than 4/5 held devices improve. Previously viewed UCI361 devices
make the whole experiment exploratory, even with this frozen run protocol.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "uci361_tinycnn_pilot.py"
ARCHIVE = Path(os.environ.get("UCI361_ARCHIVE", ROOT / "source.zip"))
spec = importlib.util.spec_from_file_location("uci361_source", SOURCE)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
data.ARCHIVE = ARCHIVE


def featurize(x, contrast):
    core = x.reshape(len(x), -1)
    if not contrast:
        return core
    paired = x[:, :, :4] - x[:, :, 4:]
    energy = np.sqrt(np.mean(x * x, axis=(1, 2), keepdims=True)) + 0.05
    # 0.5 fixed weight; standardization still fitted only to source samples.
    return np.column_stack((core, 0.5 * (paired / energy).reshape(len(x), -1)))


def evaluate(x, y, board, rank, contrast):
    prediction = np.full(len(y), -1)
    by_device = {}
    for held in range(1, 6):
        train = (board != held) & (rank >= 4)
        test = (board == held) & (rank <= 3)
        augmented = np.concatenate([x[train] * a for a in (0.5, 0.75, 1)])
        target = np.tile(y[train], 3)
        model = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))
        model.fit(featurize(augmented, contrast), target)
        prediction[test] = model.predict(featurize(x[test], contrast))
        by_device[held] = {
            "n": int(test.sum()),
            "f1": float(f1_score(y[test], prediction[test], labels=range(4), average="macro")),
            "correct": int(np.sum(y[test] == prediction[test])),
        }
    mask = prediction != -1
    mat = confusion_matrix(y[mask], prediction[mask], labels=range(4))
    return {
        "macro_f1": float(f1_score(y[mask], prediction[mask], labels=range(4), average="macro")),
        "co_recall": int(mat[0, 0]),
        "co_total": int(mat[0].sum()),
        "co_false_positive": int(mat[1:, 0].sum()),
        "confusion": mat.tolist(),
        "by_device": by_device,
        "prediction": prediction.tolist(),
    }


def main():
    x, y, board, rank, paths = data.get_trials()
    baseline = evaluate(x, y, board, rank, False)
    candidate = evaluate(x, y, board, rank, True)
    improvement = sum(candidate["by_device"][i]["f1"] > baseline["by_device"][i]["f1"] for i in range(1, 6))
    failed = (
        candidate["macro_f1"] <= baseline["macro_f1"]
        or candidate["co_recall"] < baseline["co_recall"]
        or candidate["co_false_positive"] > baseline["co_false_positive"]
        or improvement < 4
    )
    output = {"protocol": __doc__, "source_sha256": "e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167", "failed": failed, "improved_devices": improvement, "baseline": baseline, "candidate": candidate, "paths": paths}
    (ROOT / "voltage_contrast_pilot_2026_10_21.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print({"failed": failed, "baseline_f1": baseline["macro_f1"], "candidate_f1": candidate["macro_f1"], "baseline_co_recall": baseline["co_recall"], "candidate_co_recall": candidate["co_recall"], "baseline_co_fp": baseline["co_false_positive"], "candidate_co_fp": candidate["co_false_positive"]})


if __name__ == "__main__":
    main()
