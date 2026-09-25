"""Evaluate candidate person-depth gates on Bonn development sequence only.

Depth agreement between human and detector boxes is a consistency diagnostic,
not a measurement against an independently known person distance.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np
from PIL import Image

from diagnose_bonn_a2_depth_dev import ROIS, ROOT, roi_pixels
from run_bonn_a2_detector import box_iou, require_hash, sha256

FRACTIONS = (0.6, 0.8, 0.9)
MAX_IQR_M = (0.2, 0.4, 0.6)
ROI = "narrow_mid_torso_probe"


def measure(depth: np.ndarray, box: dict, roi: str = ROI) -> dict:
    region = roi_pixels(depth, box, ROIS[roi])
    values = region[region > 0].astype(np.float64) / 5000.0
    return {
        "fraction": float(values.size / region.size),
        "iqr_m": float(np.quantile(values, .75) - np.quantile(values, .25)) if values.size else None,
        "median_m": float(np.median(values)) if values.size else None,
        "pixels": int(region.size),
        "nonzero": int(values.size),
    }


def accepted(m: dict, fraction: float, max_iqr_m: float) -> bool:
    return m["iqr_m"] is not None and m["fraction"] >= fraction and m["iqr_m"] <= max_iqr_m


def main() -> None:
    freeze = json.loads((ROOT / "configs/stage8_v1a_a2_label_freeze.json").read_text(encoding="utf-8"))
    pre = json.loads((ROOT / "configs/stage8_v1a_a2_preannotation.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "data/audits/bonn_v1a_sampling_manifest.csv"
    labels_path = ROOT / "data/processed/bonn_v1a/labels/bonn_v1a_team_1_primary_rev3_assembled.json"
    zip_path = ROOT / "data/raw/bonn_rgbd/rgbd_bonn_person_tracking.zip"
    detector_path = ROOT / "outputs/stage8_a2_detector/person_detections.csv"
    for path, expected in ((manifest_path, freeze["sampling_manifest_sha256"]),
                           (labels_path, freeze["primary"]["sha256"]),
                           (zip_path, pre["sources"]["rgbd_bonn_person_tracking"])):
        require_hash(path, expected)
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        selected = [r for r in csv.DictReader(stream) if r["split"] == "development"]
    if len(selected) != 58 or any(r["sequence"] != "rgbd_bonn_person_tracking" for r in selected):
        raise ValueError("Refusing to inspect depth outside the 58 development frames")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["records"]
    with detector_path.open(encoding="utf-8", newline="") as stream:
        detections = defaultdict(list)
        for row in csv.DictReader(stream):
            if row["split"] == "development":
                detections[row["sample_id"]].append(row)
    if set(detections) - {r["sample_id"] for r in selected}:
        raise ValueError("Unexpected development detection ID")

    samples = []
    with ZipFile(zip_path) as archive:
        for row in selected:
            truths = labels[row["sample_id"]]["boxes"]
            predictions = detections[row["sample_id"]]
            if not truths:
                if predictions:
                    raise ValueError(f"Unexpected detection on empty frame: {row['sample_id']}")
                continue
            if len(truths) != 1 or len(predictions) > 1:
                raise ValueError(f"Only one person per positive frame supported: {row['sample_id']}")
            with Image.open(io.BytesIO(archive.read(row["depth_zip_member"]))) as image:
                depth = np.asarray(image)
            if depth.shape != (480, 640) or not np.issubdtype(depth.dtype, np.integer):
                raise ValueError("Unexpected depth image")
            truth = truths[0]
            gt_stats = measure(depth, truth)
            result = {"sample_id": row["sample_id"], "gt": gt_stats,
                      "gt_wide": measure(depth, truth, "wide_mid_torso_probe"),
                      "pred": None, "iou": None}
            if predictions:
                pred = predictions[0]
                pred_floats = {key: float(pred[key]) for key in ("x1", "y1", "x2", "y2")}
                iou = box_iou(list(pred_floats.values()), [truth[k] for k in ("x1", "y1", "x2", "y2")])
                if iou < freeze["one_to_one_iou_match_threshold"]:
                    raise ValueError(f"Detection does not match frozen IoU gate: {row['sample_id']}")
                # Reuse original ROI conversion: int() truncates detector's nonnegative float corners.
                result.update(pred=measure(depth, pred_floats), iou=iou)
            samples.append(result)
    if len(samples) != 32 or sum(s["pred"] is not None for s in samples) != 30:
        raise ValueError("Development frame and detector counts changed")

    grid = []
    for roi_key in ("gt", "gt_wide"):
        for fraction in FRACTIONS:
            for max_iqr_m in MAX_IQR_M:
                gt_valid = [s["sample_id"] for s in samples if accepted(s[roi_key], fraction, max_iqr_m)]
                entry = {"box_roi": roi_key, "min_nonzero_fraction": fraction,
                         "max_iqr_m": max_iqr_m, "gt_valid": len(gt_valid), "gt_unknown": 32-len(gt_valid)}
                if roi_key == "gt":
                    pred_valid = [s["sample_id"] for s in samples if s["pred"] is not None and
                                  accepted(s["pred"], fraction, max_iqr_m)]
                    entry.update(pred_valid=len(pred_valid), pred_unknown_among_32=32-len(pred_valid),
                                 gate_disagreement_among_30=sum(
                                     accepted(s["gt"], fraction, max_iqr_m) !=
                                     accepted(s["pred"], fraction, max_iqr_m)
                                     for s in samples if s["pred"] is not None))
                grid.append(entry)
    fraction, max_iqr_m = .8, .4
    mismatch = [s["sample_id"] for s in samples if s["pred"] is not None and
                accepted(s["gt"], fraction, max_iqr_m) != accepted(s["pred"], fraction, max_iqr_m)]
    both = [s for s in samples if s["pred"] is not None and
            accepted(s["gt"], fraction, max_iqr_m) and accepted(s["pred"], fraction, max_iqr_m)]
    differences = [abs(s["gt"]["median_m"]-s["pred"]["median_m"]) for s in both]
    overview = {
        "status": "development_only_candidate_gate_sensitivity_not_frozen",
        "sequence": "rgbd_bonn_person_tracking", "frames": 58, "human_person_boxes": 32,
        "matched_detector_boxes": 30, "roi_box_fractions_xyxy": ROIS[ROI],
        "wide_probe_box_fractions_xyxy": ROIS["wide_mid_torso_probe"],
        "depth_raw_units_per_meter": 5000,
        "source_hashes": {"zip": sha256(zip_path), "manifest": sha256(manifest_path),
                          "labels": sha256(labels_path), "detector_csv": sha256(detector_path),
                          "script": sha256(Path(__file__))},
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                   check=True, capture_output=True).stdout.strip(),
        "sensitivity_grid": grid,
        "candidate_0_8_0_4": {
            "gt_unknown_ids": [s["sample_id"] for s in samples if not accepted(s["gt"], fraction, max_iqr_m)],
            "pred_unknown_ids_including_missed_detections": [s["sample_id"] for s in samples if s["pred"] is None or
                                                                not accepted(s["pred"], fraction, max_iqr_m)],
            "gate_disagreement_ids_in_matched": mismatch,
            "both_valid_matched_boxes": len(both),
            "median_absolute_depth_difference_m_between_boxes": median(differences) if differences else None,
            "max_absolute_depth_difference_m_between_boxes": max(differences) if differences else None,
        },
        "limitations": "No independent true person depth; same room, temporally correlated frames. No heldout depth or physical hazard-zone evaluation."
    }
    out = ROOT / "outputs/stage8_a2_depth_gate_dev"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "frame_details.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
