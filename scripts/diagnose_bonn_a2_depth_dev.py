"""Development-sequence-only depth diagnostics; no threshold or hazard zone is fitted."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np
from PIL import Image

from run_bonn_a2_detector import ROOT, require_hash, sha256
from validate_bonn_v1a_labels import validate


# Probes, not final model choices. Bounds are fractions of a visible person box.
ROIS = {
    "full_box": (0.0, 0.0, 1.0, 1.0),
    "wide_mid_torso_probe": (0.25, 0.20, 0.75, 0.60),
    "narrow_mid_torso_probe": (0.35, 0.25, 0.65, 0.55),
}
FIELDS = ("sample_id", "roi_name", "pixel_count", "nonzero_count", "nonzero_fraction",
          "depth_median_m", "depth_p10_m", "depth_p90_m", "depth_iqr_m",
          "box_truncated", "box_touches_edge", "pair_offset_ms")


def roi_pixels(depth: np.ndarray, box: dict, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = (int(box[key]) for key in ("x1", "y1", "x2", "y2"))
    width, height = x2 - x1 + 1, y2 - y1 + 1
    left = min(639, x1 + int(np.floor(bounds[0] * width)))
    top = min(479, y1 + int(np.floor(bounds[1] * height)))
    right = min(640, x1 + max(1, int(np.ceil(bounds[2] * width))))
    bottom = min(480, y1 + max(1, int(np.ceil(bounds[3] * height))))
    if right <= left or bottom <= top:
        raise ValueError(f"Empty ROI for box {box} / {bounds}")
    return depth[top:bottom, left:right]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/bonn_v1a/labels/bonn_v1a_team_1_primary_rev3_assembled.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/audits/bonn_v1a_sampling_manifest.csv")
    parser.add_argument("--zip", type=Path, default=ROOT / "data/raw/bonn_rgbd/rgbd_bonn_person_tracking.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/stage8_a2_depth_dev")
    args = parser.parse_args()
    freeze = json.loads((ROOT / "configs/stage8_v1a_a2_label_freeze.json").read_text(encoding="utf-8"))
    pre = json.loads((ROOT / "configs/stage8_v1a_a2_preannotation.json").read_text(encoding="utf-8"))
    require_hash(args.manifest, freeze["sampling_manifest_sha256"])
    require_hash(args.labels, freeze["primary"]["sha256"])
    require_hash(args.zip, pre["sources"]["rgbd_bonn_person_tracking"])
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate(args.labels, "primary", {row["sample_id"] for row in rows})
    selected = [row for row in rows if row["split"] == "development"]
    if len(selected) != 58 or any(row["sequence"] != "rgbd_bonn_person_tracking" for row in selected):
        raise ValueError("Only 58 frames from the approved development sequence may be inspected")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["records"]
    measures = []
    with ZipFile(args.zip) as archive:
        for row in selected:
            boxes = labels[row["sample_id"]]["boxes"]
            if not boxes:
                continue
            with Image.open(io.BytesIO(archive.read(row["depth_zip_member"]))) as image:
                depth = np.asarray(image)
            if depth.shape != (480, 640) or not np.issubdtype(depth.dtype, np.integer):
                raise ValueError(f"Unexpected depth PNG: {row['sample_id']} {depth.shape} {depth.dtype}")
            for box in boxes:
                for name, bounds in ROIS.items():
                    region = roi_pixels(depth, box, bounds)
                    valid = region[region > 0].astype(float) / 5000.0
                    measures.append({
                        "sample_id": row["sample_id"], "roi_name": name,
                        "pixel_count": int(region.size), "nonzero_count": int(valid.size),
                        "nonzero_fraction": round(float(valid.size / region.size), 6),
                        "depth_median_m": round(float(np.median(valid)), 6) if valid.size else "",
                        "depth_p10_m": round(float(np.quantile(valid, 0.1)), 6) if valid.size else "",
                        "depth_p90_m": round(float(np.quantile(valid, 0.9)), 6) if valid.size else "",
                        "depth_iqr_m": round(float(np.quantile(valid, 0.75) - np.quantile(valid, 0.25)), 6) if valid.size else "",
                        "box_truncated": int(box["truncated"]),
                        "box_touches_edge": int(any(box[k] == v for k, v in
                                                    (("x1", 0), ("y1", 0), ("x2", 639), ("y2", 479)))),
                        "pair_offset_ms": row["pair_offset_ms"],
                    })
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "depth_roi_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(measures)

    overview = {"status": "development_only_diagnostic_no_final_roi_or_unknown_gate",
                "source_zip_sha256": sha256(args.zip), "primary_sha256": sha256(args.labels),
                "script_sha256": sha256(Path(__file__)),
                "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                           capture_output=True, check=True).stdout.strip(),
                "frames_in_sequence": len(selected), "person_boxes": sum(len(labels[row["sample_id"]]["boxes"]) for row in selected),
                "roi_probes": {}}
    for name in ROIS:
        subset = [x for x in measures if x["roi_name"] == name]
        fractions = [x["nonzero_fraction"] for x in subset]
        distances = [x["depth_median_m"] for x in subset if x["depth_median_m"] != ""]
        overview["roi_probes"][name] = {
            "frames_with_nonzero": len(distances),
            "fraction_nonzero_min": min(fractions),
            "fraction_nonzero_median": median(fractions),
            "fraction_nonzero_below_0_5": sum(x < 0.5 for x in fractions),
            "depth_median_m_min": min(distances) if distances else None,
            "depth_median_m_median": median(distances) if distances else None,
            "depth_median_m_max": max(distances) if distances else None,
        }
    (args.out / "diagnostic_summary.json").write_text(json.dumps(overview,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(overview,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
