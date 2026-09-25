"""Reject incomplete or malformed independent human labels before A2 scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def validate(path: Path, role: str, ids: set[str]) -> tuple[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "bonn_v1a_person_boxes_v1" or payload.get("role") != role:
        raise ValueError(f"Wrong schema or role in {path}")
    name = payload.get("annotator")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Missing human annotator identifier: {path}")
    records = payload.get("records")
    if not isinstance(records, dict) or set(records) != ids:
        raise ValueError(f"Missing or unexpected frame IDs in {path}: expected {len(ids)}, got {len(records) if isinstance(records, dict) else 'invalid'}")
    total_people = 0
    for sample_id, record in records.items():
        if not isinstance(record, dict) or record.get("reviewed") is not True:
            raise ValueError(f"Unreviewed frame {sample_id} in {path}")
        boxes = record.get("boxes")
        if not isinstance(boxes, list):
            raise ValueError(f"Missing person list: {sample_id}")
        total_people += len(boxes)
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError(f"Malformed box: {sample_id}")
            coords = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
            if not all(type(value) in (int, float) and math.isfinite(value) for value in coords):
                raise ValueError(f"Non-finite box: {sample_id}")
            x1, y1, x2, y2 = coords
            if not (0 <= x1 < x2 < 640 and 0 <= y1 < y2 < 480):
                raise ValueError(f"Box outside 640x480 image: {sample_id}")
            if any(type(box.get(attr)) is not bool for attr in ("truncated", "occluded", "uncertain")):
                raise ValueError(f"Missing box attributes: {sample_id}")
    return name, total_people


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/audits/bonn_v1a_sampling_manifest.csv"))
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    full_ids = {row["sample_id"] for row in rows}
    qc_ids = {row["sample_id"] for row in rows if row["qc_double_label"] == "1"}
    if len(full_ids) != len(rows) or len(full_ids) != 115 or len(qc_ids) != 24:
        raise ValueError("Unexpected or repeated sample IDs in manifest")
    first_name, first_people = validate(args.primary, "primary", full_ids)
    second_name, second_people = validate(args.qc, "qc", qc_ids)
    if first_name == second_name:
        raise ValueError("Independent review requires a different annotator identifier")
    print(f"Labels structurally complete: {first_name}: 115 frames/{first_people} person boxes; {second_name}: 24 frames/{second_people} person boxes")
    print("This checks structure only, not agreement or annotation correctness. Resolve disagreements before scoring.")


if __name__ == "__main__":
    main()
