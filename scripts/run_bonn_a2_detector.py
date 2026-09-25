"""Run the preselected SSDLite person detector on frozen Bonn A2 frames.

Only 2D person boxes are scored. Do not treat these indoor videos as coke-plant
images, camera calibration measurements, or intrusion ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIELDS = (
    "sample_id", "sequence", "split", "frame_index", "gt_count", "pred_count",
    "tp", "fp", "fn", "false_positive_empty_frame", "primary_box_touches_edge",
    "preprocess_inference_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)


def match_predictions(
    predictions: list[tuple[list[float], float]], truth: list[dict], threshold: float
) -> tuple[int, int, int]:
    used: set[int] = set()
    tp = 0
    for pred_box, _score in sorted(predictions, key=lambda item: -item[1]):
        candidates = [
            (box_iou(pred_box, [gt[k] for k in ("x1", "y1", "x2", "y2")]), index)
            for index, gt in enumerate(truth) if index not in used
        ]
        if candidates:
            best_iou, best_index = max(candidates)
            if best_iou >= threshold:
                used.add(best_index)
                tp += 1
    return tp, len(predictions) - tp, len(truth) - tp


def metrics(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {
        "frames": len(rows), "gt_boxes": sum(int(row["gt_count"]) for row in rows),
        "pred_boxes": sum(int(row["pred_count"]) for row in rows),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "empty_frames": sum(int(row["gt_count"]) == 0 for row in rows),
        "false_positive_empty_frames": sum(int(row["false_positive_empty_frame"]) for row in rows),
    }


def cpu_model() -> str | None:
    info = Path("/proc/cpuinfo")
    if not info.is_file():
        return platform.processor() or None
    for line in info.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/bonn_v1a/labels/bonn_v1a_team_1_primary_rev3_assembled.json")
    parser.add_argument("--qc", type=Path, default=ROOT / "data/processed/bonn_v1a/labels/bonn_v1a_team_2_qc_rev2_assembled.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "data/processed/bonn_v1a")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/audits/bonn_v1a_sampling_manifest.csv")
    parser.add_argument("--weights", type=Path, default=ROOT / "data/raw/bonn_rgbd/weights/ssdlite320_mobilenet_v3_large_coco-a79551df.pth")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/stage8_a2_detector")
    args = parser.parse_args()

    freeze = json.loads((ROOT / "configs/stage8_v1a_a2_label_freeze.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/stage8_v1a_a2_preannotation.json").read_text(encoding="utf-8"))
    require_hash(args.manifest, freeze["sampling_manifest_sha256"])
    require_hash(args.labels, freeze["primary"]["sha256"])
    require_hash(args.qc, freeze["independent_qc"]["sha256"])
    require_hash(args.weights, config["candidate_detector"]["weights_sha256"])
    from validate_bonn_v1a_labels import validate
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    all_ids = {row["sample_id"] for row in manifest}
    qc_ids = {row["sample_id"] for row in manifest if row["qc_double_label"] == "1"}
    assert len(manifest) == len(all_ids) == 115 and len(qc_ids) == 24
    validate(args.labels, "primary", all_ids)
    validate(args.qc, "qc", qc_ids)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["records"]

    # Import only after validating the provenance and dependencies. The model
    # reads verified local weights: no implicit internet downloads are allowed.
    import torch
    import torchvision
    from PIL import Image
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    if torch.__version__.split("+")[0] != config["candidate_detector"]["pinned_torch"]:
        raise ValueError(f"Unexpected torch version {torch.__version__}")
    if torchvision.__version__.split("+")[0] != config["candidate_detector"]["pinned_torchvision"]:
        raise ValueError(f"Unexpected torchvision version {torchvision.__version__}")
    torch.set_num_threads(4)
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(weights.meta["categories"])
    )
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    preprocess = weights.transforms()
    score_threshold = float(freeze["detector_score_threshold"])
    iou_threshold = float(freeze["one_to_one_iou_match_threshold"])
    person_index = int(config["candidate_detector"]["person_label_index"])

    # Warm up on development data, then measure transformation plus inference;
    # file read, matching, and CSV writing are outside the timing interval.
    first_image = args.frames / manifest[0]["local_rgb_path"]
    with Image.open(first_image) as source:
        warmup = preprocess(source.convert("RGB"))
    with torch.inference_mode():
        for _ in range(3):
            model([warmup])

    frame_results: list[dict] = []
    detection_results: list[dict] = []
    for index, row in enumerate(manifest):
        image_path = args.frames / row["local_rgb_path"]
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
        start = time.perf_counter()
        with torch.inference_mode():
            output = model([preprocess(rgb)])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        person_predictions: list[tuple[list[float], float]] = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if int(label) != person_index or float(score) < score_threshold:
                continue
            coords = [float(value) for value in box]
            value = float(score)
            person_predictions.append((coords, value))
            detection_results.append(dict(sample_id=row["sample_id"], sequence=row["sequence"],
                                          split=row["split"], score=value,
                                          x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]))
        truth = labels[row["sample_id"]]["boxes"]
        tp, fp, fn = match_predictions(person_predictions, truth, iou_threshold)
        edge_touch = any(
            box["x1"] == 0 or box["y1"] == 0 or box["x2"] == 639 or box["y2"] == 479
            for box in truth
        )
        frame_results.append(dict(
            sample_id=row["sample_id"], sequence=row["sequence"], split=row["split"],
            frame_index=row["frame_index"], gt_count=len(truth),
            pred_count=len(person_predictions), tp=tp, fp=fp, fn=fn,
            false_positive_empty_frame=int(not truth and bool(person_predictions)),
            primary_box_touches_edge=int(edge_touch), preprocess_inference_ms=round(elapsed_ms, 3),
        ))
        if (index + 1) % 20 == 0 or index + 1 == len(manifest):
            print(f"Completed {index + 1}/{len(manifest)} sampled frames", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "frame_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(frame_results)
    with (args.out / "person_detections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "sequence", "split", "score", "x1", "y1", "x2", "y2"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(detection_results)

    timings = sorted(float(row["preprocess_inference_ms"]) for row in frame_results)
    summary = {
        "status": "exploratory_indoor_sequence_2d_detection_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": metrics(frame_results),
        "sequences": {
            name: metrics([row for row in frame_results if row["sequence"] == name])
            for name in sorted({row["sequence"] for row in frame_results})
        },
        "edge_touch_sensitivity": {
            "with_edge_touch": metrics([row for row in frame_results if row["primary_box_touches_edge"]]),
            "without_edge_touch": metrics([row for row in frame_results if not row["primary_box_touches_edge"]]),
        },
        "timing_pc_cpu_preprocess_inference_ms": {
            "median": timings[len(timings) // 2], "max": max(timings), "threads": torch.get_num_threads(),
            "excludes_image_read_matching_and_output": True,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest), "primary_sha256": sha256(args.labels),
            "qc_sha256": sha256(args.qc), "weights_sha256": sha256(args.weights),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "script_sha256": sha256(Path(__file__)),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       capture_output=True, check=True).stdout.strip(),
            "person_score_threshold": score_threshold, "match_iou_threshold": iou_threshold,
            "person_label_index": person_index, "weights_enum": weights.name,
        },
        "limits": "Bonn lab recordings in similar room; person 2D boxes only. Held-out sequence annotations were seen during human QC. No real hazard-zone truth, D435 or Jetson measurement.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": summary["overall"], "sequences": summary["sequences"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
