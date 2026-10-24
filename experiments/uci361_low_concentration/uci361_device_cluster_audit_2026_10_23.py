"""Device-clustered paired audit for the frozen UCI361 SVM comparison.

This script uses saved predictions only. It does not fit or select a model.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "uci361_temporal_mlp_predictions_2026-10-19.csv"
AUG_FILE = ROOT / "uci361_three_scale_classical_predictions_2026-10-19.csv"
OUT_FILE = ROOT / "uci361_device_cluster_audit_2026_10_23.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def exact_sign_flip_pvalue(deltas: list[float], two_sided: bool) -> float:
    observed = sum(deltas) / len(deltas)
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        permuted.append(sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas))
    if two_sided:
        return sum(abs(value) >= abs(observed) - 1e-15 for value in permuted) / len(permuted)
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def main() -> None:
    raw_rows = [
        row for row in read_rows(RAW_FILE)
        if row["task"] == "low_held_board" and row["model"] == "temporal_rbf"
    ]
    aug_rows = [
        row for row in read_rows(AUG_FILE)
        if row["method"] == "temporal_rbf_svm"
    ]
    raw = {row["file"]: row for row in raw_rows}
    aug = {row["file"]: row for row in aug_rows}
    assert len(raw) == len(aug) == 192
    assert set(raw) == set(aug)

    per_device = {}
    device_deltas = []
    corrected = introduced = 0
    for device in range(1, 6):
        names = sorted(name for name, row in raw.items() if int(row["board"]) == device)
        assert names
        raw_correct = sum(raw[name]["prediction"] == raw[name]["gas"] for name in names)
        aug_correct = sum(aug[name]["prediction"] == aug[name]["truth"] for name in names)
        raw_accuracy = raw_correct / len(names)
        aug_accuracy = aug_correct / len(names)
        delta = aug_accuracy - raw_accuracy
        device_deltas.append(delta)
        per_device[str(device)] = {
            "exposure_files": len(names),
            "raw_correct": raw_correct,
            "augmented_correct": aug_correct,
            "raw_accuracy": raw_accuracy,
            "augmented_accuracy": aug_accuracy,
            "accuracy_gain": delta,
        }
        for name in names:
            truth = raw[name]["gas"]
            before = raw[name]["prediction"]
            after = aug[name]["prediction"]
            corrected += before != truth and after == truth
            introduced += before == truth and after != truth

    assert corrected == 44 and introduced == 0
    assert all(delta > 0 for delta in device_deltas)
    output = {
        "status": "Retrospective device-clustered audit of frozen saved predictions; not a new blind test.",
        "evaluation_unit": "One complete gas exposure file.",
        "independence_cluster": "Physical e-nose device (five devices).",
        "comparison": "Three-scale source-only amplitude augmentation versus the same unaugmented RBF-SVM.",
        "per_device": per_device,
        "equal_weight_mean_device_accuracy_gain": sum(device_deltas) / len(device_deltas),
        "devices_improved": sum(delta > 0 for delta in device_deltas),
        "devices_total": len(device_deltas),
        "exact_cluster_sign_flip_one_sided_p": exact_sign_flip_pvalue(device_deltas, two_sided=False),
        "exact_cluster_sign_flip_two_sided_p": exact_sign_flip_pvalue(device_deltas, two_sided=True),
        "paired_exposure_files_corrected": corrected,
        "paired_exposure_files_newly_wrong": introduced,
        "interpretation": (
            "The gain has the same direction on all five physical devices. In this retrospective audit, "
            "the one-sided exact sign-flip value is 1/32; because the directional test was added after the "
            "main result, it is descriptive rather than confirmatory. The two-sided value is 2/32 and must "
            "not be described as conventionally significant."
        ),
    }
    OUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(OUT_FILE)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
