"""Compare independent Bonn A2 annotations without changing either original JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from validate_bonn_v1a_labels import validate


ATTRIBUTES = ("truncated", "occluded", "uncertain")


def iou(first: dict, second: dict) -> float:
    left = max(first["x1"], second["x1"])
    top = max(first["y1"], second["y1"])
    right = min(first["x2"], second["x2"])
    bottom = min(first["y2"], second["y2"])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_first = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    area_second = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    return intersection / (area_first + area_second - intersection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/audits/bonn_v1a_sampling_manifest.csv"))
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    full_ids = {row["sample_id"] for row in manifest}
    qc_ids = {row["sample_id"] for row in manifest if row["qc_double_label"] == "1"}
    if len(manifest) != 115 or len(full_ids) != 115 or len(qc_ids) != 24:
        raise ValueError("Manifest must contain 115 distinct samples and 24 QC samples")
    primary_name, primary_boxes = validate(args.primary, "primary", full_ids)
    qc_name, qc_boxes = validate(args.qc, "qc", qc_ids)
    if primary_name == qc_name:
        raise ValueError("Two independent annotator identifiers required")
    primary = json.loads(args.primary.read_text(encoding="utf-8"))["records"]
    qc = json.loads(args.qc.read_text(encoding="utf-8"))["records"]
    overlaps = []
    count_disagreements = []
    attribute_disagreements = []
    agreed_empty = 0
    for sample_id in sorted(qc_ids):
        first, second = primary[sample_id]["boxes"], qc[sample_id]["boxes"]
        if len(first) != len(second):
            count_disagreements.append((sample_id, len(first), len(second)))
        elif not first:
            agreed_empty += 1
        elif len(first) == 1:
            overlaps.append((sample_id, iou(first[0], second[0])))
            if any(first[0][attr] != second[0][attr] for attr in ATTRIBUTES):
                attribute_disagreements.append(sample_id)
        else:
            # Multi-person matching requires an explicit one-to-one assignment rule.
            count_disagreements.append((sample_id, len(first), len(second)))

    print(f"Structural validation: {primary_name} {len(full_ids)} frames/{primary_boxes} boxes; "
          f"{qc_name} {len(qc_ids)} frames/{qc_boxes} boxes")
    print(f"Overlap: {len(overlaps)} single-person pairs, {agreed_empty} agreed-empty frames; "
          f"count/multiperson cases: {count_disagreements}; attribute conflicts: {attribute_disagreements}")
    if overlaps:
        scores = [score for _, score in overlaps]
        print(f"Box agreement ({len(scores)} reviewed pairs): mean IoU {mean(scores):.4f}, "
              f"median {median(scores):.4f}, minimum {min(scores):.4f}")
        print("For visual review only, pairs below 0.75 IoU (not an acceptance threshold):")
        for sample_id, score in overlaps:
            if score < 0.75:
                print(f"  {sample_id}: {score:.4f}")
    border = [(sample_id, box["x1"], box["y1"], box["x2"], box["y2"])
              for sample_id, record in primary.items() for box in record["boxes"]
              if not box["truncated"] and (box["x1"] == 0 or box["y1"] == 0
                                           or box["x2"] == 639 or box["y2"] == 479)]
    print(f"Primary boxes touching a frame edge without a truncated flag: {len(border)}")
    for item in border:
        print("  " + str(item))
    print("This audit does not adjudicate a box, infer visibility from coordinates, or score a detector.")


if __name__ == "__main__":
    main()
