"""Frozen targeted pilot before extracting UCI361 responses for this experiment.

Question: source devices only at concentration ranks 4..10; identify four
separately presented gases on a held-out device at ranks 1..3. Complete gas
exposure files are independent scoring units, never windows or augmented
copies. All source-only fitting is repeated in five leave-one-device-out folds.

Candidate mechanism: fit a concentration power exponent for each gas and
sensor channel from the source devices' high-rank median log-responses. Fit
slopes within each (device, gas, channel) to remove device intercepts, then
take the median and clip to [0.2, 1.8]. Synthesize ranks 1,2,3 from every
source file rank r by multiplying each channel by (target_rank/r)**exponent.
Train SVC C=1 on those three derived traces plus unmodified source traces.

Strong same-model control: SVC C=1 with factors (0.5,0.75,1.0), already
tested in this repository. Independent strong architecture control: frozen
prior three-factor CNN predictions under identical file/device split.
Scaler is source-only; target data cannot determine exponents or choices.
No parameter edits after target predictions. Failure of a substantive gain:
candidate macro F1 must exceed BOTH controls by >=0.01, keep CO recall at
48/48, reduce CO false positives relative to CNN's 5 and SVM's 3, and
perform no worse than the CNN on at least four of five held-out devices.
Report paired corrected/worsened file counts. With five CNN errors, even
perfect corrections would not meet a two-sided p<.05 sign test; never claim
statistical confirmation from this already-explored five-device dataset.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import uci361_tinycnn_pilot as source

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "uci361_recovered_predictions_2026-10-20.csv"


def exponents(x, y, board, rank, mask):
    """Within-device slope, then median; no target device values."""
    magnitudes = np.median(np.abs(x), axis=1)
    out = np.empty((4, 8), dtype=float)
    for gas in range(4):
        for channel in range(8):
            slopes = []
            for b in sorted(set(board[mask])):
                ix = mask & (board == b) & (y == gas)
                if len(set(rank[ix])) < 4:
                    continue
                c = np.log(rank[ix])
                r = np.log(np.maximum(magnitudes[ix, channel], 1e-4))
                slope = np.dot(c - c.mean(), r - r.mean()) / np.dot(c - c.mean(), c - c.mean())
                slopes.append(slope)
            if not slopes:
                raise ValueError((gas, channel, "insufficient source concentrations"))
            out[gas, channel] = np.clip(np.median(slopes), 0.2, 1.8)
    return out


def score(truth, pred, board, mask):
    c = confusion_matrix(truth[mask], pred[mask], labels=range(4))
    return {
        "macro_f1": float(f1_score(truth[mask], pred[mask], labels=range(4), average="macro")),
        "co_true_positive": int(c[0, 0]),
        "co_total": int(c[0].sum()),
        "co_false_positives": int(c[1:, 0].sum()),
        "confusion": c.tolist(),
        "per_device_f1": {str(k): float(f1_score(truth[mask & (board == k)], pred[mask & (board == k)], labels=range(4), average="macro", zero_division=0)) for k in range(1, 6)},
    }


def cnn_control(paths, y, board, mask):
    entries = {}
    with PREDICTIONS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "one":
                continue
            if row["file"] in entries:
                raise ValueError("Duplicate CNN file")
            entries[row["file"]] = row
    pred = np.full(len(y), -1, int)
    for i in np.flatnonzero(mask):
        row = entries[paths[i]]
        if row["truth"] != source.LABELS[y[i]] or int(row["held_devices"]) != board[i]:
            raise ValueError("CNN prediction does not match evaluation file")
        pred[i] = source.LABELS.index(row["aug_cnn"])
    return pred


def main():
    x, y, board, rank, paths = source.get_trials()  # validates official ZIP hash
    predictions = {k: np.full(len(y), -1, int) for k in ("law_svm", "fixed_svm")}
    fitted_exponents = {}
    for held in range(1, 6):
        tr = (board != held) & (rank >= 4)
        te = (board == held) & (rank <= 3)
        beta = exponents(x, y, board, rank, tr)
        fitted_exponents[str(held)] = beta.tolist()
        scaled = [x[tr]]
        source_rank = rank[tr]
        source_gas = y[tr]
        for target_rank in (1, 2, 3):
            factor = np.power(target_rank / source_rank[:, None], beta[source_gas])
            scaled.append(x[tr] * factor[:, None, :])
        specs = {
            "law_svm": (np.concatenate(scaled), np.tile(y[tr], 4)),
            "fixed_svm": (np.concatenate([x[tr] * f for f in (0.5, 0.75, 1.0)]), np.tile(y[tr], 3)),
        }
        for name, (train, labels) in specs.items():
            model = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))
            model.fit(train.reshape(len(train), -1), labels)
            predictions[name][te] = model.predict(x[te].reshape(te.sum(), -1))
        print("device", held, "test n", te.sum(), flush=True)
    mask = (rank <= 3) & (predictions["law_svm"] >= 0)
    assert int(mask.sum()) == 192
    predictions["fixed_cnn"] = cnn_control(paths, y, board, mask)
    scores = {k: score(y, v, board, mask) for k, v in predictions.items()}
    ours, svm, cnn = scores["law_svm"], scores["fixed_svm"], scores["fixed_cnn"]
    passes = (ours["macro_f1"] >= max(svm["macro_f1"], cnn["macro_f1"]) + 0.01
              and ours["co_true_positive"] == ours["co_total"] == 48
              and ours["co_false_positives"] < min(svm["co_false_positives"], cnn["co_false_positives"])
              and sum(ours["per_device_f1"][str(k)] >= cnn["per_device_f1"][str(k)] for k in range(1, 6)) >= 4)
    paired = {}
    for name in ("fixed_svm", "fixed_cnn"):
        corrected = (predictions["law_svm"][mask] == y[mask]) & (predictions[name][mask] != y[mask])
        worsened = (predictions["law_svm"][mask] != y[mask]) & (predictions[name][mask] == y[mask])
        paired[name] = {"corrected": int(corrected.sum()), "worsened": int(worsened.sum())}
    output = {"protocol": __doc__, "dataset": "https://doi.org/10.24432/C5MW3K", "source_sha256": "e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167", "exponents_by_held_device": fitted_exponents, "scores": scores, "paired": paired, "passes_substantive_gain": bool(passes), "not_independent_blind_test": True}
    (ROOT / "uci361_concentration_law_2026_10_22.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    with (ROOT / "uci361_concentration_law_2026_10_22.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["file", "device", "concentration_rank", "truth", *predictions])
        for i in np.flatnonzero(mask):
            w.writerow([paths[i], board[i], rank[i], source.LABELS[y[i]], *(source.LABELS[p[i]] for p in predictions.values())])
    print({k: (v["macro_f1"], v["co_true_positive"], v["co_false_positives"]) for k, v in scores.items()})
    print("paired", paired, "passes", passes)


if __name__ == "__main__":
    main()
