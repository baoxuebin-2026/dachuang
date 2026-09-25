"""Compare virtual camera-relative region definitions on development frames only.

These candidate boundaries are illustrative geometry, not surveyed industrial
zones. No second-sequence depth or hazard truth is read by this script.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

from run_bonn_a2_detector import ROOT, require_hash, sha256


def lateral(x_m: float, boundary_m: float = .35) -> str:
    return "inside" if x_m >= boundary_m else "near" if x_m >= boundary_m - .20 else "outside"


def forward(z_m: float, boundary_m: float = 1.70) -> str:
    return "inside" if z_m <= boundary_m else "near" if z_m <= boundary_m + .15 else "outside"


def main() -> None:
    freeze_path = ROOT / "configs/stage8_v1a_a2_depth_gate_freeze.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["status"] != "approved_depth_validity_gate_frozen_before_second_sequence_depth_inspection":
        raise ValueError("Depth gate is not frozen")
    if frozen["roi_fraction_xyxy_of_person_box"] != [.35, .25, .65, .55]:
        raise ValueError("Unexpected region of interest")
    if frozen["validity_gate"]["min_nonzero_fraction_inclusive"] != .8 or frozen["validity_gate"]["max_iqr_m_inclusive"] != .2:
        raise ValueError("Unexpected approved depth gate")

    detector_path = ROOT / "outputs/stage8_a2_detector/person_detections.csv"
    detail_path = ROOT / "outputs/stage8_a2_depth_gate_dev/frame_details.json"
    gate_summary_path = ROOT / "outputs/stage8_a2_depth_gate_dev/summary.json"
    manifest_path = ROOT / "data/audits/bonn_v1a_sampling_manifest.csv"
    camera = json.loads((ROOT / "configs/stage8_v1a_a2_preannotation.json").read_text(encoding="utf-8"))["depth_rule"]["camera_intrinsics"]
    if camera != {"fx": 542.822841, "fy": 542.57687, "cx": 315.59352, "cy": 237.756098}:
        raise ValueError("Unexpected camera calibration")
    require_hash(detector_path, frozen["detector_predictions_sha256"])
    require_hash(manifest_path, frozen["sampling_manifest_sha256"])
    gate_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
    if gate_summary["source_hashes"]["detector_csv"] != sha256(detector_path) or gate_summary["human_person_boxes"] != 32:
        raise ValueError("Development gate results have changed")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if len(detail) != 32 or any(r["sample_id"].split(":")[0] != "rgbd_bonn_person_tracking" for r in detail):
        raise ValueError("Refusing to evaluate non-development depth detail")
    by_id = {r["sample_id"]: r for r in detail}
    if len(by_id) != len(detail):
        raise ValueError("Duplicate frame detail")
    with detector_path.open(encoding="utf-8", newline="") as stream:
        detector_rows = {r["sample_id"]: r for r in csv.DictReader(stream) if r["split"] == "development"}
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        selected = [r for r in csv.DictReader(stream) if r["split"] == "development"]
    if len(selected) != 58 or any(r["sequence"] != "rgbd_bonn_person_tracking" for r in selected):
        raise ValueError("Development sequence mismatch")
    if set(by_id) - {r["sample_id"] for r in selected} or set(detector_rows) - set(by_id):
        raise ValueError("Person-box ID mismatch")
    approved = []
    counts = {name: Counter() for name in ("lateral", "forward")}
    for row in selected:
        key = row["sample_id"]
        if key not in by_id:
            continue
        frame = by_id[key]
        pred = frame["pred"]
        detection = detector_rows.get(key)
        result = {"sample_id": key, "lateral": "unknown", "forward": "unknown", "x_m": None, "z_m": None,
                  "reason": None}
        if detection is None or pred is None:
            result["reason"] = "missed_detection"
        elif pred["nonzero"] == 0 or pred["fraction"] < .8 or pred["iqr_m"] > .2:
            result["reason"] = "depth_gate_unknown"
        else:
            z_m = pred["median_m"]
            u_px = (float(detection["x1"]) + float(detection["x2"])) / 2
            # Optical camera coordinate X points right; point is the box's torso
            # center proxy, not the person's ground position or body boundary.
            x_m = (u_px - camera["cx"]) * z_m / camera["fx"]
            result.update(x_m=x_m, z_m=z_m, lateral=lateral(x_m), forward=forward(z_m))
        for name in counts:
            counts[name][result[name]] += 1
        approved.append(result)
    if len(approved) != 32 or sum(x["reason"] is None for x in approved) != 29:
        raise ValueError("The approved gate should yield 29 valid of 32 labeled person frames")
    valid_points = [x for x in approved if x["reason"] is None]
    boundary_sensitivity = {
        "lateral": {str(b): dict(Counter(lateral(r["x_m"], b) for r in valid_points)) for b in (.30, .35, .40)},
        "forward": {str(b): dict(Counter(forward(r["z_m"], b) for r in valid_points)) for b in (1.65, 1.70, 1.75)},
    }
    summary = {
        "status": "development_only_virtual_region_candidates_not_approved",
        "sequence": "rgbd_bonn_person_tracking", "sampled_frames": 58,
        "person_frames": len(approved), "valid_person_depth_frames": 29,
        "region_definitions": {
            "lateral": "X=(box center x - cx)*torso ROI median depth/fx; inside X>=0.35m; near 0.15<=X<0.35m; outside X<0.15m",
            "forward": "Z=torso ROI median depth; inside Z<=1.70m; near 1.70<Z<=1.85m; outside Z>1.85m",
            "unknown": "Missing detection or failed approved A1 depth gate; empty human-ground-truth frames are excluded",
        },
        "coordinate_source": "Official Bonn camera intrinsics in configs/stage8_v1a_a2_preannotation.json",
        "counts": {name: dict(counter) for name, counter in counts.items()},
        "development_boundary_sensitivity_valid_only": boundary_sensitivity,
        "inputs_sha256": {"freeze": sha256(freeze_path), "detector": sha256(detector_path),
                          "dev_gate_summary": sha256(gate_summary_path), "dev_frame_detail": sha256(detail_path),
                          "manifest": sha256(manifest_path), "script": sha256(Path(__file__))},
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                   check=True, capture_output=True).stdout.strip(),
        "limits": "Illustrative camera-relative torso center. No surveyed plant hazard zone, person footprint, depth truth, or second-sequence depth opened."
    }
    out = ROOT / "outputs/stage8_a2_virtual_region_dev"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "frame_details.json").write_text(json.dumps(approved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
