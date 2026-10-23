"""Retrospective target-label budget control, rules frozen before this run.

UCI361 complete exposure is the unit. Leave each device out: source devices
ranks 4..10 only; target ranks 1..3 are never used for fitting. For an
optional target calibration comparator, draw five target *high-rank* trials
per class (20 labeled gas exposures) using seeds 21/37/59, disjoint from low
test trials. SVC C=1, gamma=scale, source-only train StandardScaler for source
arms; target-only scaler for target-only arm. All augmentation uses fixed
response factors (0.5, 0.75, 1), no target low labels or test tuning.

Arms: source-only augmented SVM; source + target20 raw SVM; source + target20
augmented SVM; target20-only augmented SVM. The 20 target trials are a label
budget, not new devices. A meaningful no-calibration advantage would require
the source-only method to exceed every target20 arm by >=0.02 macro F1 in all
three seeds with CO recall no lower. Report any reversal and CO errors. The
dataset and augmentation factors were inspected previously; this is a
retrospective same-dataset control, never an independent blind test.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import uci361_tinycnn_pilot as source


ROOT = Path(__file__).resolve().parent
SCALES = (0.5, 0.75, 1.0)


def predict(train_x, train_y, test_x, augment):
    if augment:
        train_y = np.tile(train_y, len(SCALES))
        train_x = np.concatenate([train_x * a for a in SCALES])
    model = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))
    model.fit(train_x.reshape(len(train_x), -1), train_y)
    return model.predict(test_x.reshape(len(test_x), -1))


def main():
    x, y, boards, ranks, names = source.get_trials()
    runs = {}
    for seed in (21, 37, 59):
        rng = np.random.default_rng(seed)
        pred = {name: np.full(len(y), -1) for name in (
            "source_augmented", "source_plus_target20_raw",
            "source_plus_target20_augmented", "target20_augmented")}
        pilot_names = {}
        for held in range(1, 6):
            src = np.where((boards != held) & (ranks >= 4))[0]
            target_hi = np.where((boards == held) & (ranks >= 4))[0]
            test = np.where((boards == held) & (ranks <= 3))[0]
            pilot = np.sort(np.concatenate([
                rng.choice(target_hi[y[target_hi] == cls], size=5, replace=False)
                for cls in range(4)]))
            assert not set(pilot).intersection(test)
            pilot_names[str(held)] = [names[i] for i in pilot]
            all_train = np.concatenate([src, pilot])
            pred["source_augmented"][test] = predict(x[src], y[src], x[test], True)
            pred["source_plus_target20_raw"][test] = predict(x[all_train], y[all_train], x[test], False)
            pred["source_plus_target20_augmented"][test] = predict(x[all_train], y[all_train], x[test], True)
            pred["target20_augmented"][test] = predict(x[pilot], y[pilot], x[test], True)
        runs[str(seed)] = {"pilot_files": pilot_names, "methods": {}}
        for method, guesses in pred.items():
            sel = guesses >= 0
            assert sel.sum() == 192 == len({names[i] for i in np.where(sel)[0]})
            mat = confusion_matrix(y[sel], guesses[sel], labels=range(4))
            runs[str(seed)]["methods"][method] = {
                "macro_f1": float(f1_score(y[sel], guesses[sel], labels=range(4), average="macro")),
                "correct": int(np.sum(y[sel] == guesses[sel])),
                "co_correct": int(mat[0, 0]),
                "non_co_as_co": int(mat[1:, 0].sum()),
                "per_board_macro_f1": {str(b): float(f1_score(
                    y[(boards == b) & sel], guesses[(boards == b) & sel],
                    labels=range(4), average="macro")) for b in range(1, 6)},
            }
        print(seed, {k: {m: round(v[m], 4) if m == "macro_f1" else v[m]
                           for m in ("macro_f1", "correct", "co_correct", "non_co_as_co")}
                      for k, v in runs[str(seed)]["methods"].items()}, flush=True)
    winner = all(
        all(
            vals["source_augmented"]["macro_f1"] >= vals[k]["macro_f1"] + 0.02
            and vals["source_augmented"]["co_correct"] >= vals[k]["co_correct"]
            for k in vals if k != "source_augmented")
        for vals in (runs[str(seed)]["methods"] for seed in (21, 37, 59)))
    result = {"scope": "Retrospective UCI361 control", "frozen_seeds": [21, 37, 59],
              "per_class_target_calibration": 5, "runs": runs,
              "meaningful_no_label_advantage": winner}
    (ROOT / "uci361_target20_calibration_control_2026_10_22.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
