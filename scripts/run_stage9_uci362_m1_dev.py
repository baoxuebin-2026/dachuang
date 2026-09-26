#!/usr/bin/env python3
"""Confirmed M1 date-level threshold comparison on UCI 362 development IDs only.

This is a post hoc, exploratory development experiment. No test record is scored.
The thresholds and evaluation rules were fixed in the M1 protocol before running.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_stage9_uci362_locked as locked


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage9_uci362/m1_dev_run"
PROTOCOL = ROOT / "docs/stage9-m1-date-balanced-dev-protocol.md"
METHODS = ("pooled_frozen", "date_050", "date_075_primary", "date_090")


def read_development() -> tuple[dict[int, dict], dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    if locked.sha256(locked.ARCHIVE) != locked.SOURCE_SHA:
        raise ValueError("UCI 362 source SHA changed")
    if locked.sha256(locked.MANIFEST) != locked.MANIFEST_SHA:
        raise ValueError("Split manifest SHA changed")
    manifest = json.loads(locked.MANIFEST.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != locked.SOURCE_SHA:
        raise ValueError("Manifest source SHA changed")
    dev = {int(x["id"]): x for x in manifest["trials"] if x["split"] in ("train", "validation")}
    test_ids = {int(x["id"]) for x in manifest["trials"] if x["split"] == "test"}
    if len(dev) != 78 or len(test_ids) != 21 or set(dev) & test_ids:
        raise ValueError("Unexpected split")
    run = json.loads((locked.OUT / "run.json").read_text(encoding="utf-8"))
    if (run["manifest_sha256"] != locked.MANIFEST_SHA or
            run["source_sha256"] != locked.SOURCE_SHA or
            run["code_sha256"] != locked.sha256(Path(locked.__file__))):
        raise ValueError("Frozen model changed")

    with zipfile.ZipFile(locked.ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("HT_Sensor_dataset.zip"))) as inner:
            parts = pd.read_csv(inner.open("HT_Sensor_dataset.dat"), sep=r"\s+", chunksize=50000)
            selected = []
            for part in parts:
                if list(part.columns) != locked.COLUMNS:
                    raise ValueError("Unexpected UCI 362 columns")
                selected.append(part.loc[part["id"].isin(dev), locked.COLUMNS])
    frame = pd.concat(selected, ignore_index=True)
    if set(frame["id"].unique()) != set(dev) or set(frame["id"].unique()) & test_ids:
        raise ValueError("Development-only read failed")
    frame["second"] = np.floor(frame["time"].to_numpy(dtype=float) * 3600).astype(int)
    grouped = frame.groupby(["id", "second"], sort=True)[locked.SENSOR_COLS].median()
    arrays = {}
    for ident, part in grouped.groupby(level="id", sort=True):
        seconds = part.index.get_level_values("second").to_numpy(dtype=int)
        values = part.to_numpy(dtype=float)
        if np.any(np.diff(seconds) < 1):
            raise ValueError(f"Non-monotonic seconds in {ident}")
        arrays[int(ident)] = (seconds, values)
    return dev, arrays, run


def fit_thresholds(dev: dict[int, dict], arrays: dict, original: dict) -> tuple[dict, list[dict], dict]:
    scale = np.asarray(original["calibration"]["scale"])
    scores_by_day = defaultdict(list)
    for ident, info in dev.items():
        if info["split"] != "train":
            continue
        seconds, values = arrays[ident]
        center, valid = locked.calibrate(seconds, values)
        if center is None:
            continue
        bg = valid & (seconds >= -2700) & (seconds < -300)
        if bg.any():
            scores_by_day[info["date"]].append(
                np.median(np.abs((values[bg] - center) / scale), axis=1)
            )
    day_scores = {day: np.concatenate(parts) for day, parts in scores_by_day.items()}
    if len(day_scores) != original["calibration"]["training_dates"]:
        raise ValueError("Training date count differs from frozen run")
    total = sum(len(x) for x in day_scores.values())
    if total != original["calibration"]["training_background_bins"]:
        raise ValueError("Training background count differs from frozen run")
    all_scores = np.concatenate(list(day_scores.values()))
    frozen = original["calibration"]["thresholds"]["median_8"]
    if not np.isclose(np.quantile(all_scores, 0.999), frozen, rtol=0, atol=1e-10):
        raise ValueError("Original pooled threshold failed independent reconstruction")

    daily = [{"date": day, "valid_background_bins": len(values),
              "q_0999": float(np.quantile(values, 0.999)),
              "pooled_tail_bins": int(np.sum(values > frozen))}
             for day, values in sorted(day_scores.items())]
    day_q = np.asarray([x["q_0999"] for x in daily])
    thresholds = {"pooled_frozen": float(frozen),
                  "date_050": float(np.quantile(day_q, 0.50)),
                  "date_075_primary": float(np.quantile(day_q, 0.75)),
                  "date_090": float(np.quantile(day_q, 0.90))}
    excluded = "09-03-15"
    if excluded not in day_scores:
        raise ValueError("Diagnostic date missing")
    without = np.asarray([x["q_0999"] for x in daily if x["date"] != excluded])
    sensitivity = {
        "excluded_date": excluded,
        "excluded_q_0999": float(np.quantile(day_scores[excluded], 0.999)),
        "q_0999_rank_ascending": int(1 + np.sum(day_q < np.quantile(day_scores[excluded], 0.999))),
        "date_075_without_date": float(np.quantile(without, 0.75)),
        "pooled_0999_without_date": float(np.quantile(
            np.concatenate([x for day, x in day_scores.items() if day != excluded]), 0.999)),
        "note": "Diagnostic only; all original dates remain in primary threshold and evaluation.",
    }
    return thresholds, daily, sensitivity


def evaluate(dev: dict[int, dict], arrays: dict, original: dict, thresholds: dict) -> tuple[list, list, dict, list]:
    scale = np.asarray(original["calibration"]["scale"])
    by_date = defaultdict(list)
    for info in dev.values():
        by_date[info["date"]].append(info)
    pure_dates = {date for date, entries in by_date.items()
                  if all(x["class"] == "background" for x in entries)}
    if len(pure_dates) != 8 or any(len(by_date[day]) != 1 for day in pure_dates):
        raise ValueError("Unexpected pure-background day structure; observation hours need review")

    records, events = [], []
    for ident, info in sorted(dev.items()):
        seconds, values = arrays[ident]
        center, valid = locked.calibrate(seconds, values)
        post_valid = int((valid & (seconds >= 0) & (seconds < 300)).sum())
        baseline_bins = int((valid & (seconds >= -3600) & (seconds < -2700)).sum())
        negative_bins = int((valid & (seconds >= -2700) & (seconds < -60)).sum())
        pure_bg = info["date"] in pure_dates
        bg_bins = int((valid & (seconds >= -2700)).sum()) if pure_bg else None
        for method in METHODS:
            status = ("unknown_calibration" if center is None else
                      "unknown_post_coverage" if info["class"] != "background" and post_valid < 120
                      else "eligible")
            found = [] if center is None else locked.g2_events(
                seconds, values, valid, center, scale, thresholds[method], "median_8"
            )
            positive = [e for e in found if e["origin_bin_s"] >= 0 and 0 < e["completed_s"] <= 300]
            before = [e for e in found if -2700 < e["completed_s"] <= -60]
            background = [e for e in found if e["completed_s"] > -2700] if pure_bg else []
            detected = int(bool(positive)) if status == "eligible" and info["class"] != "background" else None
            records.append({
                "id": ident, "date": info["date"], "split": info["split"], "class": info["class"],
                "method": method, "threshold": thresholds[method], "status": status,
                "pure_background_day": int(pure_bg), "baseline_valid_bins": baseline_bins,
                "post5_valid_bins": post_valid, "pre_negative_valid_bins": negative_bins,
                "background_observed_bins": bg_bins, "detected_within_5min": detected,
                "first_detection_s": positive[0]["completed_s"] if detected else None,
                "pre_event_count": len(before),
                "background_g2_count": len(background) if pure_bg and center is not None else None,
            })
            for event in found:
                events.append({"id": ident, "date": info["date"], "split": info["split"],
                               "class": info["class"], "method": method, **event,
                               "scope": ("pure_background" if pure_bg else
                                         "stimulus_5min" if event in positive else
                                         "pre_stimulus" if event in before else "other")})

    summaries = {}
    per_pure_day = []
    for split in ("train", "validation"):
        summaries[split] = {}
        for method in METHODS:
            chosen = [r for r in records if r["split"] == split and r["method"] == method]
            positives = [r for r in chosen if r["class"] != "background"]
            eligible = [r for r in positives if r["status"] == "eligible"]
            detected = [r for r in eligible if r["detected_within_5min"]]
            bg = [r for r in chosen if r["pure_background_day"]]
            eligible_bg = [r for r in bg if r["status"] == "eligible"]
            summaries[split][method] = {
                "stimulus_total": len(positives), "stimulus_eligible": len(eligible),
                "stimulus_unknown": dict(Counter(r["status"] for r in positives if r["status"] != "eligible")),
                "detected_within_5min": len(detected),
                "detected_by_class": {
                    label: {"numerator": sum(r["detected_within_5min"] == 1 for r in eligible if r["class"] == label),
                            "denominator": sum(r["class"] == label for r in eligible)}
                    for label in ("wine", "banana")},
                "median_delay_s_detected_only": float(np.median([r["first_detection_s"] for r in detected])) if detected else None,
                "pre_stimulus_trigger_records": sum(r["pre_event_count"] > 0 for r in eligible),
                "pre_stimulus_g2_events": sum(r["pre_event_count"] for r in eligible),
                "pure_background_days": len(bg),
                "pure_background_eligible_days": len(eligible_bg),
                "pure_background_triggered_days": sum(r["background_g2_count"] > 0 for r in eligible_bg),
                "pure_background_g2_events": sum(r["background_g2_count"] for r in eligible_bg),
                "pure_background_observed_hours": sum(r["background_observed_bins"] for r in eligible_bg) / 3600,
            }
            for r in bg:
                matching = [e for e in events if e["id"] == r["id"] and e["method"] == method and
                            e["scope"] == "pure_background"]
                per_pure_day.append({"date": r["date"], "split": split, "id": r["id"],
                                     "method": method, "status": r["status"],
                                     "observed_hours": (r["background_observed_bins"] or 0) / 3600,
                                     "g2_events": r["background_g2_count"],
                                     "first_g2_s": matching[0]["completed_s"] if matching else None})
    return records, events, summaries, per_pure_day


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError("M1 development run already exists; do not overwrite")
    locked.self_check()
    dev, arrays, original = read_development()
    thresholds, daily, sensitivity = fit_thresholds(dev, arrays, original)
    records, events, summary, pure_day = evaluate(dev, arrays, original, thresholds)
    for split in ("train", "validation"):
        actual = summary[split]["pooled_frozen"]
        expected = original["summary"][split]["median_8"]
        if (actual["detected_within_5min"] != expected["detected_within_5min"] or
                actual["pure_background_g2_events"] != expected["background_g2_events"] or
                not np.isclose(actual["pure_background_observed_hours"], expected["background_observed_hours"])):
            raise ValueError(f"Frozen baseline reconstruction failed in {split}")
    OUT.mkdir(parents=True, exist_ok=False)
    csv_write(OUT / "per_record.csv", records)
    csv_write(OUT / "events.csv", events,
              ["id", "date", "split", "class", "method", "completed_s", "origin_bin_s", "score_at_trigger", "scope"])
    csv_write(OUT / "daily_training_thresholds.csv", daily)
    csv_write(OUT / "pure_background_days.csv", pure_day)
    report = {
        "label": "post_hoc_development_only_m1_confirmed_protocol",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": locked.sha256(PROTOCOL),
        "source_sha256": locked.SOURCE_SHA, "manifest_sha256": locked.MANIFEST_SHA,
        "code_sha256": locked.sha256(Path(__file__)),
        "frozen_run_code_sha256": original["code_sha256"],
        "retained_development_ids": len(dev), "test_ids_scored": 0,
        "training_dates_with_background": len(daily),
        "thresholds": thresholds, "excluded_day_sensitivity_diagnostic_only": sensitivity,
        "summary": summary,
    }
    (OUT / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
