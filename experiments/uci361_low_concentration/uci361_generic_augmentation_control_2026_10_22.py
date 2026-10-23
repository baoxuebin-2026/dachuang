"""Frozen before first execution: retrospective UCI361 augmentation specificity audit.

Scoring unit: one whole exposure file, 192 unique held-device low-rank files.
Each fold trains on other devices' ranks 4--10 only. Identical 20x8 log-
response feature extraction and tiny CNN architecture/seed for every arm.
Three training views for each original source trial:

* amplitude: factors (0.5, 0.75, 1.0), the existing paper candidate;
* smooth_noise: two independent temporally smoothed Gaussian perturbations,
  each scaled per trial to the same Frobenius displacement as amplitude
  factors 0.5 and 0.75, followed by the unaltered trial;
* temporal_shift: one-bin left/right shifts with replicated edge values,
  followed by the unaltered trial.

The amplitude-specificity claim fails if either generic arm attains at least
the amplitude arm's pooled macro F1, CO recall, and no more CO false positives.
Report each held-device score and paired corrected/new errors. This is not a
blind independent test; the five-device dataset was already inspected.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix, f1_score

import uci361_tinycnn_pilot as cnn


ROOT = Path(__file__).resolve().parent
ARMS = ("amplitude", "smooth_noise", "temporal_shift")


def make_views(x, arm, held):
    if arm == "amplitude":
        return np.concatenate((x * 0.5, x * 0.75, x), axis=0)
    if arm == "temporal_shift":
        left = np.concatenate((x[:, 1:, :], x[:, -1:, :]), axis=1)
        right = np.concatenate((x[:, :1, :], x[:, :-1, :]), axis=1)
        return np.concatenate((left, right, x), axis=0)
    if arm == "smooth_noise":
        rng = np.random.default_rng(20261022 + held)
        out = []
        signal_norm = np.linalg.norm(x.reshape(len(x), -1), axis=1)
        for displacement in (0.5, 0.25):
            z = gaussian_filter1d(rng.normal(size=x.shape), sigma=2, axis=1)
            z_norm = np.linalg.norm(z.reshape(len(z), -1), axis=1)
            z *= (displacement * signal_norm / np.maximum(z_norm, 1e-12))[:, None, None]
            out.append(x + z)
        return np.concatenate((*out, x), axis=0)
    raise ValueError(arm)


def main():
    x, y, boards, ranks, paths = cnn.get_trials()
    all_results = {}
    predictions = {}
    for arm in ARMS:
        pred = np.full(len(y), -1, dtype=int)
        fit = {}
        board_f1 = {}
        for held in range(1, 6):
            train = (boards != held) & (ranks >= 4)
            test = (boards == held) & (ranks <= 3)
            train_x = make_views(x[train], arm, held)
            train_y = np.tile(y[train], 3)
            p, fit_info = cnn.train_predict(train_x, train_y, x[test], seed=2026)
            pred[test] = p
            fit[str(held)] = fit_info
            board_f1[str(held)] = float(f1_score(y[test], p, labels=range(4), average="macro"))
            print(arm, held, "F1", board_f1[str(held)], flush=True)
        selected = pred >= 0
        assert selected.sum() == 192
        assert len(set(paths[i] for i in np.flatnonzero(selected))) == 192
        cm = confusion_matrix(y[selected], pred[selected], labels=range(4))
        predictions[arm] = pred
        all_results[arm] = {
            "n_unique_test_files": 192,
            "macro_f1": float(f1_score(y[selected], pred[selected], labels=range(4), average="macro")),
            "correct": int((y[selected] == pred[selected]).sum()),
            "co_correct": int(cm[0, 0]),
            "co_total": int(cm[0].sum()),
            "non_co_as_co": int(cm[1:, 0].sum()),
            "confusion": cm.tolist(),
            "per_board_macro_f1": board_f1,
            "fitting": fit,
        }
    selected = predictions["amplitude"] >= 0
    amp_ok = predictions["amplitude"][selected] == y[selected]
    pairwise = {}
    for arm in ARMS[1:]:
        generic_ok = predictions[arm][selected] == y[selected]
        pairwise[arm] = {
            "fixed_by_amplitude": int((~generic_ok & amp_ok).sum()),
            "new_amplitude_errors": int((generic_ok & ~amp_ok).sum()),
        }
    amp = all_results["amplitude"]
    decisions = {}
    for arm in ARMS[1:]:
        other = all_results[arm]
        decisions[arm] = (
            other["macro_f1"] >= amp["macro_f1"]
            and other["co_correct"] >= amp["co_correct"]
            and other["non_co_as_co"] <= amp["non_co_as_co"]
        )
    dest = ROOT / "uci361_generic_augmentation_control_2026_10_22.json"
    dest.write_text(json.dumps({
        "scope": "Retrospective, same previously inspected UCI361 archive; not independent validation.",
        "source_sha256": "e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167",
        "failure_rule": "Generic arm F1 >= amplitude F1, CO correct >= amplitude, and false CO <= amplitude.",
        "arms": all_results,
        "paired": pairwise,
        "amplitude_specificity_refuted_by_generic": decisions,
    }, ensure_ascii=False, indent=2) + "\n")
    print("RESULT", {k: {q: v[q] for q in ("macro_f1", "correct", "co_correct", "non_co_as_co")} for k, v in all_results.items()}, flush=True)
    print("SAVED", dest, flush=True)


if __name__ == "__main__":
    main()
