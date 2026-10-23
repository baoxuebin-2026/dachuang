"""Retrospective same-SVM normalization control on the verified UCI361 ZIP.

Frozen before viewing this control's results: a complete gas-exposure file is
the scoring unit; leave one entire device out, train only on the other devices'
ranks 4-10, test ranks 1-3. StandardScaler and SVC(C=1, gamma='scale') are fit
on source devices only. Compare unaugmented log responses, three-scale response
augmentation (0.5, 0.75, 1), and per-file RMS-normalized responses with the
same stabilizer 0.05 as the existing CNN control. No target labels, calibration
data, or parameter choices enter training. This is not a new blind dataset.

The amplitude claim loses its strong-baseline advantage if normalized SVM
achieves at least its four-class macro F1 without a higher CO miss count.
Report all outcomes, including a negative result, without retuning.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import uci361_tinycnn_pilot as source
from paper_preflight_2026_10_22 import LABELS, summarize


ROOT = Path(__file__).resolve().parent


def normalize(x):
    rms = np.sqrt(np.mean(x ** 2, axis=(1, 2), keepdims=True))
    return x / np.sqrt(rms ** 2 + 0.05 ** 2)


def predict(x, y, boards, ranks, method):
    predictions = np.full(len(y), -1)
    for held in range(1, 6):
        train = (boards != held) & (ranks >= 4)
        test = (boards == held) & (ranks <= 3)
        if method == "normalized":
            features = normalize(x)
            xx, yy, target = features[train], y[train], features[test]
        elif method == "three_scale":
            xx = np.concatenate([x[train] * factor for factor in (0.5, 0.75, 1)])
            yy = np.tile(y[train], 3)
            target = x[test]
        else:
            xx, yy, target = x[train], y[train], x[test]
        model = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))
        model.fit(xx.reshape(len(xx), -1), yy)
        predictions[test] = model.predict(target.reshape(int(test.sum()), -1))
    return predictions


def main(path):
    source.ARCHIVE = Path(path)
    x, y, boards, ranks, files = source.get_trials()
    mask = ranks <= 3
    names = [files[i] for i in np.flatnonzero(mask)]
    truth = [LABELS[y[i]] for i in np.flatnonzero(mask)]
    output = {}
    rows = {name: {"file": name, "truth": label} for name, label in zip(names, truth)}
    for method in ("raw", "normalized", "three_scale"):
        pred = predict(x, y, boards, ranks, method)
        predicted = [LABELS[pred[i]] for i in np.flatnonzero(mask)]
        output[method] = summarize(names, truth, predicted)
        for name, label in zip(names, predicted):
            rows[name][method] = label
        print(method, output[method]["macro_f1"], output[method]["correct"], flush=True)
    assert len(names) == len(set(names)) == 192
    assert {k: sum(r["truth"] == k for r in rows.values()) for k in LABELS} == {k: 48 for k in LABELS}

    earlier_raw = {r["file"]: r["prediction"] for r in csv.DictReader(
        (ROOT / "uci361_temporal_mlp_predictions_2026-10-19.csv").open(newline=""))
        if r["task"] == "low_held_board" and r["model"] == "temporal_rbf"}
    earlier_aug = {r["file"]: r["prediction"] for r in csv.DictReader(
        (ROOT / "uci361_three_scale_classical_predictions_2026-10-19.csv").open(newline=""))
        if r["method"] == "temporal_rbf_svm"}
    assert all(row["raw"] == earlier_raw[name] and row["three_scale"] == earlier_aug[name]
               for name, row in rows.items()), "Reproduction differs from saved predictions"
    result = {
        "status": "Retrospective normalized SVM control on previously inspected UCI361 device folds.",
        "protocol": __doc__,
        "source_sha256": source.hashlib.sha256(source.ARCHIVE.read_bytes()).hexdigest(),
        "methods": output,
        "normalized_to_three_scale_paired": {
            "corrected_by_three_scale": sum(
                row["normalized"] != row["truth"] and row["three_scale"] == row["truth"]
                for row in rows.values()
            ),
            "new_errors_with_three_scale": sum(
                row["normalized"] == row["truth"] and row["three_scale"] != row["truth"]
                for row in rows.values()
            ),
        },
        "decision": "Augmentation retains four-class and CO-miss advantage" if (
            output["three_scale"]["macro_f1"] > output["normalized"]["macro_f1"]
            and output["three_scale"]["confusion_matrix"][0][0]
            >= output["normalized"]["confusion_matrix"][0][0]
        ) else "No four-class plus CO-miss advantage over normalized SVM",
    }
    (ROOT / "paper_normalization_control_2026_10_22.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    with (ROOT / "paper_normalization_control_2026_10_22.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "truth", "raw", "normalized", "three_scale"))
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))
    print(result["decision"])


if __name__ == "__main__":
    main(sys.argv[1])
