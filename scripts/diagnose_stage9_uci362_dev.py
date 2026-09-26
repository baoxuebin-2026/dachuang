#!/usr/bin/env python3
"""Post hoc diagnosis on UCI 362 train/validation dates only.

Reads the frozen model but fits no new threshold and does not calculate test
scores, test events, or new model performance. Output is diagnostic evidence.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_stage9_uci362_locked as locked


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage9_uci362/dev_diagnostic.json"


def max_consecutive(seconds: np.ndarray, above: np.ndarray) -> int:
    longest = length = 0
    previous = None
    for second, high in zip(seconds, above):
        length = length + 1 if high and (previous is None or second == previous + 1) else (1 if high else 0)
        longest = max(longest, length)
        previous = int(second)
    return longest


def main() -> None:
    manifest = json.loads(locked.MANIFEST.read_text(encoding="utf-8"))
    if locked.sha256(locked.MANIFEST) != locked.MANIFEST_SHA:
        raise ValueError("Split manifest changed")
    dev = {int(x["id"]): x for x in manifest["trials"] if x["split"] in ("train", "validation")}
    test_ids = {int(x["id"]) for x in manifest["trials"] if x["split"] == "test"}
    run = json.loads((locked.OUT / "run.json").read_text(encoding="utf-8"))
    if run["manifest_sha256"] != locked.MANIFEST_SHA or run["source_sha256"] != locked.SOURCE_SHA:
        raise ValueError("Run does not match the agreed data")
    if locked.sha256(locked.ARCHIVE) != locked.SOURCE_SHA:
        raise ValueError("Unexpected original ZIP")

    with zipfile.ZipFile(locked.ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("HT_Sensor_dataset.zip"))) as inner:
            chunks = pd.read_csv(inner.open("HT_Sensor_dataset.dat"), sep=r"\s+", chunksize=50000)
            dev_chunks = [part.loc[part["id"].isin(dev), locked.COLUMNS] for part in chunks]
    frame = pd.concat(dev_chunks, ignore_index=True)
    if set(frame["id"].unique()) != set(dev) or set(frame["id"].unique()) & test_ids:
        raise ValueError("Dev-only source selection failed")
    frame["second"] = np.floor(frame["time"].to_numpy(dtype=float) * 3600).astype(int)
    grouped = frame.groupby(["id", "second"], sort=True)[locked.SENSOR_COLS].median()
    scale = np.asarray(run["calibration"]["scale"])
    threshold = run["calibration"]["thresholds"]["median_8"]
    threshold_r1 = run["calibration"]["thresholds"]["R1"]
    per_record = []
    training_tail_by_day = Counter()
    training_tail_by_id = Counter()
    total_training_background = 0

    for ident, part in grouped.groupby(level="id", sort=True):
        ident = int(ident)
        info = dev[ident]
        seconds = part.index.get_level_values("second").to_numpy(dtype=int)
        values = part.to_numpy(dtype=float)
        center, valid = locked.calibrate(seconds, values)
        record = {"id": ident, "date": info["date"], "split": info["split"],
                  "class": info["class"], "calibration_eligible": center is not None}
        if center is None:
            per_record.append(record)
            continue

        z = np.abs((values - center) / scale)
        score = np.median(z, axis=1)
        if info["split"] == "train":
            background = valid & (seconds >= -2700) & (seconds < -300)
            total_training_background += int(background.sum())
            training_tail_by_day[info["date"]] += int(np.sum(score[background] > threshold))
            training_tail_by_id[str(ident)] += int(np.sum(score[background] > threshold))

        first5 = valid & (seconds >= 0) & (seconds < 300)
        pre5 = valid & (seconds >= -300) & (seconds < -60)
        record["post5_valid_bins"] = int(first5.sum())
        record["pre5_valid_bins"] = int(pre5.sum())
        if info["class"] == "background" or record["post5_valid_bins"] < 120:
            per_record.append(record)
            continue

        mask = (seconds >= 0) & (seconds < 300)
        highs = valid[mask] & (score[mask] > threshold)
        highs_r1 = valid[mask] & (z[mask, 0] > threshold_r1)
        record.update({
            "score_peak5": float(np.max(score[first5])),
            "peak_to_threshold": float(np.max(score[first5]) / threshold),
            "median_score_pre5": float(np.median(score[pre5])) if pre5.sum() >= 120 else None,
            "median_score_post5": float(np.median(score[first5])),
            "g1_bins_first5": int(highs.sum()),
            "max_consecutive_g1": max_consecutive(seconds[mask], highs),
            "r1_g1_bins_first5": int(highs_r1.sum()),
            "r1_max_consecutive_g1": max_consecutive(seconds[mask], highs_r1),
            "channels_ever_above_own_train_scale_by_3": int(np.sum(np.any(z[first5] > 3, axis=0))),
            "median_abs_z_post_by_channel": [float(x) for x in np.median(z[first5], axis=0)],
        })
        per_record.append(record)

    positives = [r for r in per_record if r["class"] != "background" and
                 r.get("post5_valid_bins", 0) >= 120 and r["calibration_eligible"]]
    summary = {}
    for split in ("train", "validation"):
        for label in ("wine", "banana"):
            subset = [r for r in positives if r["split"] == split and r["class"] == label]
            summary[f"{split}:{label}"] = {
                "records": len(subset),
                "peak_ratio_median": float(np.median([r["peak_to_threshold"] for r in subset])),
                "peak_ratio_range": [float(min(r["peak_to_threshold"] for r in subset)),
                                     float(max(r["peak_to_threshold"] for r in subset))],
                "with_any_g1_bin": sum(r["g1_bins_first5"] > 0 for r in subset),
                "with_three_consecutive_g1": sum(r["max_consecutive_g1"] >= 3 for r in subset),
                "with_any_r1_g1_bin": sum(r["r1_g1_bins_first5"] > 0 for r in subset),
                "with_three_consecutive_r1_g1": sum(r["r1_max_consecutive_g1"] >= 3 for r in subset),
                "median_channels_ever_above_3scale": float(np.median([r["channels_ever_above_own_train_scale_by_3"] for r in subset])),
                "post_channel_median_abs_z": [float(v) for v in np.median(
                    np.asarray([r["median_abs_z_post_by_channel"] for r in subset]), axis=0)],
            }
    result = {
        "label": "post_hoc_development_only_no_new_model",
        "source_sha256": locked.SOURCE_SHA,
        "manifest_sha256": locked.MANIFEST_SHA,
        "locked_run_code_sha256": run["code_sha256"],
        "diagnostic_code_sha256": locked.sha256(Path(__file__)),
        "tested_ids_read_for_scores": 0,
        "development_id_count": len(dev),
        "training_background_bins_recomputed": total_training_background,
        "training_tail_above_frozen_threshold_by_day": dict(sorted(training_tail_by_day.items())),
        "training_tail_above_frozen_threshold_by_id": dict(sorted(training_tail_by_id.items(), key=lambda x: int(x[0]))),
        "summary": summary,
        "per_record": per_record,
    }
    if total_training_background != run["calibration"]["training_background_bins"]:
        raise ValueError("Training background count changed")
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_record"}, indent=2))


if __name__ == "__main__":
    main()
