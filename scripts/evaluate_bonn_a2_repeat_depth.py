"""One-pass indoor repeat-sequence check after A1 and G1 were frozen.

Scores depth availability and virtual camera-relative categories only; there
is no true person depth or ground-truth hazard-zone label in this dataset.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np
from PIL import Image

from compare_bonn_a2_virtual_regions_dev import lateral
from evaluate_bonn_a2_depth_gate_dev import accepted, measure
from run_bonn_a2_detector import ROOT, box_iou, require_hash, sha256
from validate_bonn_v1a_labels import validate


def point(box: dict, measured: dict, cx: float, fx: float) -> dict:
    z_m = measured["median_m"]
    u_px = (float(box["x1"]) + float(box["x2"])) / 2
    x_m = (u_px - cx) * z_m / fx
    return {"z_m": z_m, "x_m": x_m, "region": lateral(x_m),
            "nearest_boundary_m": min(abs(x_m - .15), abs(x_m - .35))}


def main() -> None:
    gate_path = ROOT / "configs/stage8_v1a_a2_depth_gate_freeze.json"
    region_path = ROOT / "configs/stage8_v1a_a2_virtual_region_freeze.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    region = json.loads(region_path.read_text(encoding="utf-8"))
    pre = json.loads((ROOT / "configs/stage8_v1a_a2_preannotation.json").read_text(encoding="utf-8"))
    frozen_labels = json.loads((ROOT / "configs/stage8_v1a_a2_label_freeze.json").read_text(encoding="utf-8"))
    if region["status"] != "approved_G1_camera_relative_region_frozen_before_second_sequence_depth_inspection":
        raise ValueError("G1 freeze is required before opening the repeat sequence")
    require_hash(gate_path, region["depth_gate_freeze_sha256"])
    if gate["validity_gate"]["min_nonzero_fraction_inclusive"] != .8 or gate["validity_gate"]["max_iqr_m_inclusive"] != .2:
        raise ValueError("Unapproved A1 gate")
    if gate["roi_fraction_xyxy_of_person_box"] != [.35, .25, .65, .55]:
        raise ValueError("Unapproved ROI")
    if region["region_categories"]["inside"] != "X >= 0.35 m" or region["region_categories"]["near"] != "0.15 m <= X < 0.35 m":
        raise ValueError("Unapproved G1 boundary")
    if region["development_selection_commit"] != "d199c0984615e2a0318889c0e18e039efc7b155c":
        raise ValueError("Development decision source changed")

    manifest_path = ROOT / "data/audits/bonn_v1a_sampling_manifest.csv"
    label_path = ROOT / "data/processed/bonn_v1a/labels/bonn_v1a_team_1_primary_rev3_assembled.json"
    detector_path = ROOT / "outputs/stage8_a2_detector/person_detections.csv"
    zip_path = ROOT / "data/raw/bonn_rgbd/rgbd_bonn_person_tracking2.zip"
    require_hash(manifest_path, gate["sampling_manifest_sha256"])
    require_hash(label_path, gate["primary_labels_sha256"])
    require_hash(detector_path, gate["detector_predictions_sha256"])
    require_hash(zip_path, pre["sources"][region["repeat_sequence"]])
    camera = pre["depth_rule"]["camera_intrinsics"]
    if camera != {"fx": 542.822841, "fy": 542.57687, "cx": 315.59352, "cy": 237.756098}:
        raise ValueError("Unexpected camera calibration")

    with manifest_path.open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    validate(label_path, "primary", {row["sample_id"] for row in manifest})
    rows = [r for r in manifest if r["sequence"] == region["repeat_sequence"] and r["split"] == "held_out_similar_room"]
    if len(rows) != 57 or len({r["sample_id"] for r in rows}) != 57:
        raise ValueError("Expected 57 fixed repeat-sequence frames")
    labels = json.loads(label_path.read_text(encoding="utf-8"))["records"]
    with detector_path.open(encoding="utf-8", newline="") as stream:
        predictions = defaultdict(list)
        for r in csv.DictReader(stream):
            if r["sequence"] == region["repeat_sequence"]:
                if r["split"] != "held_out_similar_room" or float(r["score"]) < frozen_labels["detector_score_threshold"]:
                    raise ValueError("Frozen detector predictions changed")
                predictions[r["sample_id"]].append(r)
    if set(predictions) - {r["sample_id"] for r in rows}:
        raise ValueError("Unexpected detector frame")

    evaluated = []
    with ZipFile(zip_path) as archive:
        for row in rows:
            key = row["sample_id"]
            truth, detected = labels[key]["boxes"], predictions[key]
            if len(truth) > 1 or len(detected) > 1:
                raise ValueError(f"Only one labeled person supported at {key}")
            frame = {"sample_id": key, "frame_index": int(row["frame_index"]),
                     "pair_offset_ms": float(row["pair_offset_ms"]),
                     "has_human_person": bool(truth), "has_detector_person": bool(detected),
                     "state": "no_detected_person", "unknown_reason": None,
                     "gt_valid": None, "pred_valid": None, "gt": None, "pred": None,
                     "gt_point": None, "pred_point": None, "box_iou": None}
            if not truth:
                if detected:
                    raise ValueError(f"Unexpected detector output in empty frame {key}")
                evaluated.append(frame)
                continue
            with Image.open(io.BytesIO(archive.read(row["depth_zip_member"]))) as img:
                depth = np.asarray(img)
            if depth.shape != (480, 640) or not np.issubdtype(depth.dtype, np.integer):
                raise ValueError(f"Unexpected depth PNG for {key}")
            gt_measure = measure(depth, truth[0])
            gt_valid = accepted(gt_measure, .8, .2)
            frame.update(gt=gt_measure, gt_valid=gt_valid)
            if gt_valid:
                frame["gt_point"] = point(truth[0], gt_measure, camera["cx"], camera["fx"])
            if not detected:
                frame.update(state="unknown", unknown_reason="missed_2d_detection")
                evaluated.append(frame)
                continue
            pred_box = {k: float(detected[0][k]) for k in ("x1", "y1", "x2", "y2")}
            overlap = box_iou([pred_box[k] for k in ("x1", "y1", "x2", "y2")],
                              [truth[0][k] for k in ("x1", "y1", "x2", "y2")])
            if overlap < frozen_labels["one_to_one_iou_match_threshold"]:
                raise ValueError(f"Detection no longer matches the frozen 2D label at {key}")
            pred_measure = measure(depth, pred_box)
            pred_valid = accepted(pred_measure, .8, .2)
            frame.update(box_iou=overlap, pred=pred_measure, pred_valid=pred_valid)
            if pred_valid:
                pred_point = point(pred_box, pred_measure, camera["cx"], camera["fx"])
                frame.update(pred_point=pred_point, state=pred_point["region"])
            else:
                frame.update(state="unknown", unknown_reason=("no_depth_pixels" if pred_measure["nonzero"] == 0
                         else "too_many_missing_pixels" if pred_measure["fraction"] < .8 else "depth_iqr_over_0_2_m"))
            evaluated.append(frame)

    positive = [r for r in evaluated if r["has_human_person"]]
    empty = [r for r in evaluated if not r["has_human_person"]]
    if len(evaluated) != 57 or len(positive) != 34 or len(empty) != 23 or sum(r["has_detector_person"] for r in positive) != 32:
        raise ValueError("Frozen 2D denominators changed")
    both_valid = [r for r in positive if r["gt_valid"] and r["pred_valid"]]
    valid_pred = [r for r in positive if r["pred_valid"]]
    depth_difference = [abs(r["pred_point"]["z_m"] - r["gt_point"]["z_m"]) for r in both_valid]
    x_difference = [abs(r["pred_point"]["x_m"] - r["gt_point"]["x_m"]) for r in both_valid]
    summary = {
        "status": "one_pass_similar_room_repeat_after_A1_and_G1_freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sequence": region["repeat_sequence"], "frames": len(evaluated),
        "human_person_frames": len(positive), "empty_human_label_frames": len(empty),
        "detector_box_in_positive_frames": sum(r["has_detector_person"] for r in positive),
        "gt_box_valid_depth": sum(r["gt_valid"] for r in positive),
        "detector_box_valid_depth": len(valid_pred),
        "region_states_over_34_person_frames": dict(Counter(r["state"] for r in positive)),
        "unknown_reasons": dict(Counter(r["unknown_reason"] for r in positive if r["state"] == "unknown")),
        "gt_pred_gate_disagreement_ids_among_32_matched": [r["sample_id"] for r in positive if r["has_detector_person"] and r["gt_valid"] != r["pred_valid"]],
        "both_valid_matched_pairs": len(both_valid),
        "region_disagreement_ids_among_both_valid": [r["sample_id"] for r in both_valid if r["pred_point"]["region"] != r["gt_point"]["region"]],
        "median_absolute_box_depth_difference_m": median(depth_difference) if depth_difference else None,
        "median_absolute_box_x_difference_m": median(x_difference) if x_difference else None,
        "valid_pred_nearest_boundary_at_most_0_05_m": sum(r["pred_point"]["nearest_boundary_m"] <= .05 for r in valid_pred),
        "near_boundary_ids": [r["sample_id"] for r in valid_pred if r["pred_point"]["nearest_boundary_m"] <= .05],
        "provenance": {
            "git_head_before_run": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
            "region_freeze_sha256": sha256(region_path), "depth_gate_freeze_sha256": sha256(gate_path),
            "source_zip_sha256": sha256(zip_path), "manifest_sha256": sha256(manifest_path),
            "human_labels_sha256": sha256(label_path), "detector_predictions_sha256": sha256(detector_path),
            "script_sha256": sha256(Path(__file__))
        },
        "limits": "Same-room indoor repeat, previously visible during label QC. Box agreement is not independent depth accuracy; virtual G1 boundary has no real-world intrusion ground truth. No D435, Jetson or coke-plant test."
    }
    out = ROOT / "outputs/stage8_a2_repeat_depth"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "frame_details.json").write_text(json.dumps(evaluated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
