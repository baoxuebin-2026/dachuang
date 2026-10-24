"""Audit the already-recorded UCI361 paper comparisons without refitting models.

This is a retrospective consistency check. It does not create a new test set.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABELS = ("CO", "Me", "Ea", "Ey")


def read_rows(filename: str) -> list[dict[str, str]]:
    with (ROOT / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def macro_f1(truth: list[str], predicted: list[str]) -> float:
    scores = []
    for label in LABELS:
        tp = sum(y == label and p == label for y, p in zip(truth, predicted))
        fp = sum(y != label and p == label for y, p in zip(truth, predicted))
        fn = sum(y == label and p != label for y, p in zip(truth, predicted))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def summarize(files: list[str], truth: list[str], predicted: list[str]) -> dict:
    board = [int(Path(name).name[1]) for name in files]
    matrix = [
        [sum(y == actual and p == chosen for y, p in zip(truth, predicted))
         for chosen in LABELS]
        for actual in LABELS
    ]
    return {
        "macro_f1": macro_f1(truth, predicted),
        "correct": sum(y == p for y, p in zip(truth, predicted)),
        "confusion_labels": list(LABELS),
        "confusion_matrix": matrix,
        "non_co_as_co": sum(y != "CO" and p == "CO" for y, p in zip(truth, predicted)),
        "per_board_macro_f1": {
            str(b): macro_f1(
                [y for y, bb in zip(truth, board) if bb == b],
                [p for p, bb in zip(predicted, board) if bb == b],
            )
            for b in range(1, 6)
        },
    }


def index_unique(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result = {row["file"]: row for row in rows}
    assert len(rows) == len(result), f"Duplicate files in {name}"
    return result


def main() -> None:
    raw = index_unique(
        [row for row in read_rows("uci361_temporal_mlp_predictions_2026-10-19.csv")
         if row["task"] == "low_held_board" and row["model"] == "temporal_rbf"],
        "raw SVM",
    )
    augmented = index_unique(
        [row for row in read_rows("uci361_three_scale_classical_predictions_2026-10-19.csv")
         if row["method"] == "temporal_rbf_svm"],
        "augmented SVM",
    )
    recovered = index_unique(
        [row for row in read_rows("uci361_recovered_predictions_2026-10-20.csv")
         if row["split"] == "one"],
        "recovered one-device predictions",
    )
    assert set(raw) == set(augmented) == set(recovered)
    assert len(raw) == 192
    files = sorted(raw)
    truth = [raw[name]["gas"] for name in files]
    board_counts = Counter()
    for name, label in zip(files, truth):
        board = int(Path(name).name[1])
        rank = int(Path(name).stem.split("_F")[1].split("_")[0]) // 10
        assert label in LABELS and 1 <= rank <= 3
        assert int(raw[name]["board"]) == int(augmented[name]["board"]) == board
        assert int(augmented[name]["level"]) == int(raw[name]["level"]) == rank
        assert augmented[name]["truth"] == recovered[name]["truth"] == label
        assert int(recovered[name]["held_devices"]) == board
        assert recovered[name]["aug_svm"] == augmented[name]["prediction"]
        board_counts[board] += 1
    assert Counter(truth) == {label: 48 for label in LABELS}
    assert board_counts == {1: 48, 2: 48, 3: 48, 4: 24, 5: 24}

    predictions = {
        "svm_no_augmentation": [raw[name]["prediction"] for name in files],
        "svm_three_scale": [augmented[name]["prediction"] for name in files],
        "cnn_no_augmentation": [recovered[name]["raw_cnn"] for name in files],
        "cnn_per_record_rms": [recovered[name]["norm_cnn"] for name in files],
        "cnn_three_scale": [recovered[name]["aug_cnn"] for name in files],
    }
    results = {key: summarize(files, truth, pred) for key, pred in predictions.items()}
    old = json.loads((ROOT / "uci361_temporal_mlp_exploratory_2026-10-19.json").read_text())
    new = json.loads((ROOT / "uci361_three_scale_classical_baselines_2026-10-19.json").read_text())
    assert abs(results["svm_no_augmentation"]["macro_f1"] - old["results"]["temporal_rbf_low_held_board"]["macro_f1"]) < 1e-12
    assert abs(results["svm_three_scale"]["macro_f1"] - new["results"]["temporal_rbf_svm"]["macro_f1"]) < 1e-12
    assert results["svm_three_scale"]["confusion_matrix"] == new["results"]["temporal_rbf_svm"]["confusion"]

    before = predictions["svm_no_augmentation"]
    after = predictions["svm_three_scale"]
    paired = {
        "corrected_by_augmentation": sum(a != y and b == y for a, b, y in zip(before, after, truth)),
        "new_errors_after_augmentation": sum(a == y and b != y for a, b, y in zip(before, after, truth)),
        "both_correct": sum(a == y and b == y for a, b, y in zip(before, after, truth)),
        "both_wrong": sum(a != y and b != y for a, b, y in zip(before, after, truth)),
    }
    assert sum(paired.values()) == 192
    normalized = predictions["cnn_per_record_rms"]
    against_normalized = {
        "normalized_cnn_errors_fixed_by_augmented_svm": sum(
            a != y and b == y for a, b, y in zip(normalized, after, truth)
        ),
        "new_errors_relative_to_normalized_cnn": sum(
            a == y and b != y for a, b, y in zip(normalized, after, truth)
        ),
    }
    out = {
        "status": "Retrospective audit of previously inspected UCI361 observations; no new blind test.",
        "paper_task": "Four pure-gas classes on an unseen board at concentration ranks 1-3; source boards use ranks 4-10.",
        "source_dataset": "UCI 361 Twin gas sensor arrays; 640 physical trials, five boards.",
        "scoring_unit": "One complete exposure file; 192 unique held-board test files.",
        "class_counts": dict(Counter(truth)),
        "board_counts": {str(k): board_counts[k] for k in range(1, 6)},
        "methods": results,
        "paired_svm_correctness": paired,
        "augmented_svm_vs_normalized_cnn_paired": against_normalized,
        "paper_claim_boundary": "Factors were selected after earlier UCI361 outcomes; no mixture recognition or prospective external replication is claimed.",
        "comparison_note": "Per-record RMS-normalized CNN is a close four-class F1 baseline; it misses 6/48 CO trials, while the augmented SVM misses 0/48. It must not be omitted from the manuscript comparison.",
        "resolved_after_this_audit": [
            "paper_normalization_control_2026_10_22.py reran the verified archive, reproduced raw and three-scale predictions exactly, and added the same-SVM RMS-normalized control.",
            "The RMS-normalized SVM reached macro-F1 0.917774 and missed 8/48 CO files; the three-scale SVM retained macro-F1 0.963233 and missed 0/48 CO files.",
        ],
        "pending_before_submission": [
            "Confirm author names, project funding number, and journal formatting from actual editorial instructions.",
        ],
        "provenance": [
            "uci361_temporal_mlp_predictions_2026-10-19.csv",
            "uci361_three_scale_classical_predictions_2026-10-19.csv",
            "uci361_recovered_predictions_2026-10-20.csv",
        ],
    }
    target = ROOT / "paper_preflight_2026_10_22.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(target)
    for key, value in results.items():
        print(key, f"F1={value['macro_f1']:.6f}", f"correct={value['correct']}/192", f"nonCO_as_CO={value['non_co_as_co']}")
    print("paired SVM", paired)


if __name__ == "__main__":
    main()
