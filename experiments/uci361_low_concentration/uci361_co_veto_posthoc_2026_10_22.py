"""Retrospective CO-specific arbitration on the already inspected UCI361 pilot.

The rule was conceived after looking at held-device low-concentration errors.
Its score is exploratory and cannot count as an independent or blind test.
For an unseen future dataset, freeze a full classifier and a concentration-law
classifier using source devices only. Keep the full classifier's prediction
except when it calls CO and the law classifier calls a different gas; in that
case use the law classifier's gas. A strong negative control applies precisely
the same arbitration using the fixed-amplitude SVM. The independent-data
failure criterion is an improvement of at least 0.01 in macro F1 over the
stronger of the full classifier and fixed-amplitude control, unchanged CO
recall, and fewer false CO alerts, with no worse per-device F1 on 4/5 devices.
Never silently retune the rule using this retrospective dataset.
"""

import csv
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent


def evaluate(rows, predictions):
    labels = ["CO", "Me", "Ea", "Ey"]
    truth = [r["truth"] for r in rows]
    matrix = confusion_matrix(truth, predictions, labels=labels)
    return {
        "macro_f1": float(f1_score(truth, predictions, labels=labels, average="macro")),
        "correct": sum(a == b for a, b in zip(truth, predictions)),
        "co_recall_numerator": int(matrix[0, 0]),
        "co_recall_denominator": int(matrix[0].sum()),
        "non_co_called_co": int(matrix[1:, 0].sum()),
        "per_device_f1": {
            str(d): float(f1_score(
                [r["truth"] for r in rows if int(r["device"]) == d],
                [p for r, p in zip(rows, predictions) if int(r["device"]) == d],
                labels=labels, average="macro", zero_division=0))
            for d in range(1, 6)
        },
        "confusion": matrix.tolist(),
    }


def veto(rows, challenger):
    return [r[challenger] if r["fixed_cnn"] == "CO" and r[challenger] != "CO"
            else r["fixed_cnn"] for r in rows]


def main():
    with (ROOT / "uci361_concentration_law_2026_10_22.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 192 and len({r["file"] for r in rows}) == 192
    labels = {key: [r[key] for r in rows] for key in ("fixed_cnn", "fixed_svm", "law_svm")}
    labels["law_co_veto"] = veto(rows, "law_svm")
    labels["fixed_co_veto"] = veto(rows, "fixed_svm")
    results = {name: evaluate(rows, pred) for name, pred in labels.items()}
    changed = {name: [r["file"] for r, p in zip(rows, labels[name])
                      if r["fixed_cnn"] != p] for name in ("law_co_veto", "fixed_co_veto")}
    outcome = {
        "protocol": __doc__, "dataset": "https://doi.org/10.24432/C5MW3K",
        "test_exposures": 192, "already_inspected_test_labels": True,
        "independent_confirmation": False, "results": results,
        "changed_files": changed,
    }
    (ROOT / "uci361_co_veto_posthoc_2026_10_22.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
    print({k: (v["macro_f1"], v["correct"], v["non_co_called_co"])
           for k, v in results.items()})


if __name__ == "__main__":
    main()
