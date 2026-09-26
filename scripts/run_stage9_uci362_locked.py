#!/usr/bin/env python3
"""Execute the confirmed UCI 362 date-held-out gas-response protocol exactly once.

python scripts/run_stage9_uci362_locked.py
python scripts/run_stage9_uci362_locked.py --self-check

The benchmark is household wine/banana stimulation, never a toxic gas hazard
or a coke-plant incident. This script never tunes a threshold on validation/test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/uci362/uci362_original.zip"
MANIFEST = ROOT / "results/stage9_uci362/proposed_split_manifest.json"
OUT = ROOT / "results/stage9_uci362/locked_run"
SOURCE_SHA = "7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395"
MANIFEST_SHA = "cd5aa93c00dbf7efb2ee61fe69c94d0acae9e64497e93bf76df94c15e37cfb26"
METHODS = ("median_8", "R1")
SENSOR_COLS = [f"R{i}" for i in range(1, 9)]
COLUMNS = ["id", "time", *SENSOR_COLS, "Temp.", "Humidity"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load() -> tuple[dict[int, dict], dict[int, tuple[np.ndarray, np.ndarray]]]:
    if sha256(ARCHIVE) != SOURCE_SHA:
        raise ValueError("Original archive SHA-256 differs from frozen protocol")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha256(MANIFEST) != MANIFEST_SHA or manifest["source_sha256"] != SOURCE_SHA or manifest["seed"] != "20260926":
        raise ValueError("Unexpected split manifest")
    entries = {int(r["id"]): r for r in manifest["trials"]}
    with zipfile.ZipFile(ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("HT_Sensor_dataset.zip"))) as inner:
            frame = pd.read_csv(inner.open("HT_Sensor_dataset.dat"), sep=r"\s+")
    if list(frame.columns) != COLUMNS or len(frame) != 928991:
        raise ValueError(f"Unanticipated source fields or row count: {list(frame.columns)} {len(frame)}")
    if set(frame["id"].unique()) != set(entries):
        raise ValueError("Time-series IDs do not match the locked manifest")

    frame["second"] = np.floor(frame["time"].to_numpy(dtype=float) * 3600).astype(int)
    grouped = frame.groupby(["id", "second"], sort=True)[SENSOR_COLS].median()
    arrays = {}
    for ident, part in grouped.groupby(level="id", sort=True):
        seconds = part.index.get_level_values("second").to_numpy(dtype=int)
        values = part.to_numpy(dtype=float)
        if len(np.unique(seconds)) != len(seconds) or np.any(np.diff(seconds) < 1):
            raise ValueError(f"Second bins not strictly increasing for ID {ident}")
        arrays[int(ident)] = (seconds, values)
    return entries, arrays


def calibrate(seconds: np.ndarray, values: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    base = valid & (seconds >= -3600) & (seconds < -2700)
    if int(base.sum()) < 600:
        return None, valid
    return np.median(values[base], axis=0), valid


def fit(entries: dict[int, dict], arrays: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict:
    residuals = []
    centers = []
    training_dates = set()
    for ident, info in entries.items():
        if info["split"] != "train":
            continue
        seconds, values = arrays[ident]
        center, valid = calibrate(seconds, values)
        if center is None:
            continue
        centers.append(center)
        bg = valid & (seconds >= -2700) & (seconds < -300)
        if np.any(bg):
            residuals.append(values[bg] - center)
            training_dates.add(info["date"])
    if len(training_dates) < 10 or sum(len(x) for x in residuals) < 1200:
        raise ValueError("Training background does not meet the frozen calibration minimum")

    center_global = np.median(np.stack(centers), axis=0)
    deviations = np.concatenate(residuals)
    median_residual = np.median(deviations, axis=0)
    mad = 1.4826 * np.median(np.abs(deviations - median_residual), axis=0)
    scale = np.maximum(mad, np.maximum(0.001 * np.abs(center_global), 0.001))
    multi_train = np.median(np.abs(deviations / scale), axis=1)
    r1_train = np.abs(deviations[:, 0] / scale[0])
    return {
        "scale": scale.tolist(), "training_center_median": center_global.tolist(),
        "training_dates": len(training_dates), "training_background_bins": len(deviations),
        "thresholds": {"median_8": float(np.quantile(multi_train, 0.999)),
                       "R1": float(np.quantile(r1_train, 0.999))},
    }


def g2_events(seconds: np.ndarray, values: np.ndarray, valid: np.ndarray,
              center: np.ndarray, scale: np.ndarray, threshold: float,
              method: str) -> list[dict]:
    """Return only rising G2 transitions; prior seconds never carry across gaps."""
    score = np.median(np.abs((values - center) / scale), axis=1) if method == "median_8" else np.abs((values[:, 0] - center[0]) / scale[0])
    run_length = 0
    run_start = None
    previous_second = None
    events = []
    for n, second in enumerate(seconds):
        if second < -2700:
            continue
        if previous_second is not None and second != previous_second + 1:
            run_length = 0
            run_start = None
        if not valid[n] or not np.isfinite(score[n]) or score[n] <= threshold:
            run_length = 0
            run_start = None
        else:
            if run_length == 0:
                run_start = int(second)
            run_length += 1
            if run_length == 3:
                events.append({"completed_s": int(second) + 1,
                               "origin_bin_s": run_start, "score_at_trigger": float(score[n])})
        previous_second = int(second)
    return events


def evaluate(entries: dict[int, dict], arrays: dict[int, tuple[np.ndarray, np.ndarray]], fit_info: dict) -> tuple[list[dict], list[dict], dict]:
    records = []
    event_rows = []
    dates = defaultdict(list)
    for info in entries.values():
        dates[info["date"]].append(info)
    background_only = {day for day, members in dates.items()
                       if all(x["class"] == "background" for x in members)}
    scale = np.asarray(fit_info["scale"])

    for ident, info in sorted(entries.items()):
        seconds, values = arrays[ident]
        center, valid = calibrate(seconds, values)
        post_valid = int((valid & (seconds >= 0) & (seconds < 300)).sum())
        negative_valid = int((valid & (seconds >= -2700) & (seconds < -60)).sum())
        bg_valid = int((valid & (seconds >= -2700)).sum()) if info["date"] in background_only else None
        primary_negative = info["date"] in background_only
        for method in METHODS:
            status = ("unknown_calibration" if center is None else
                      "unknown_post_coverage" if info["class"] != "background" and post_valid < 120 else "eligible")
            events = [] if center is None else g2_events(seconds, values, valid, center, scale,
                                                        fit_info["thresholds"][method], method)
            positive = [e for e in events if e["origin_bin_s"] >= 0 and 0 < e["completed_s"] <= 300]
            before = [e for e in events if -2700 < e["completed_s"] <= -60]
            seeded = [e for e in events if e["origin_bin_s"] < 0 and 0 < e["completed_s"] <= 300]
            background = [e for e in events if e["completed_s"] > -2700] if primary_negative else []
            first = positive[0]["completed_s"] if positive else None
            # A missing post-onset window cannot count as a negative test outcome.
            detected = int(bool(positive)) if status == "eligible" and info["class"] != "background" else None
            records.append({
                "id": ident, "date": info["date"], "split": info["split"],
                "class": info["class"], "method": method,
                "status": status, "primary_background_day": int(primary_negative),
                "baseline_valid_bins": int((valid & (seconds >= -3600) & (seconds < -2700)).sum()),
                "post5_valid_bins": post_valid, "pre_negative_valid_bins": negative_valid,
                "background_observed_bins": bg_valid, "detected_within_5min": detected,
                "first_detection_s": first if detected else None,
                "pre_event_count": len(before), "seeded_event_count": len(seeded),
                "background_g2_count": len(background) if primary_negative and center is not None else None,
            })
            for e in events:
                event_rows.append({"id": ident, "date": info["date"], "split": info["split"],
                                   "class": info["class"], "method": method, **e,
                                   "scope": ("primary_background" if primary_negative else
                                             "stimulus_response" if e in positive else
                                             "pre_stimulus" if e in before else
                                             "seeded_after_onset" if e in seeded else "other")})

    summaries = {}
    for split in ("train", "validation", "test"):
        summaries[split] = {}
        for method in METHODS:
            selected = [r for r in records if r["split"] == split and r["method"] == method]
            positives = [r for r in selected if r["class"] != "background"]
            eligible = [r for r in positives if r["status"] == "eligible"]
            bg = [r for r in selected if r["primary_background_day"]]
            eligible_bg = [r for r in bg if r["status"] == "eligible"]
            detected = [r for r in eligible if r["detected_within_5min"]]
            summaries[split][method] = {
                "stimulus_records_total": len(positives),
                "stimulus_eligible": len(eligible),
                "stimulus_unknown": dict(Counter(r["status"] for r in positives if r["status"] != "eligible")),
                "detected_within_5min": len(detected),
                "detected_by_class": {
                    label: {"numerator": sum(r["detected_within_5min"] == 1 for r in eligible if r["class"] == label),
                            "denominator": sum(r["class"] == label for r in eligible)}
                    for label in ("wine", "banana")},
                "median_delay_s_detected_only": float(np.median([r["first_detection_s"] for r in detected])) if detected else None,
                "pre_stimulus_trigger_records": sum(bool(r["pre_event_count"]) for r in eligible),
                "background_only_days_total": len({r["date"] for r in bg}),
                "background_only_days_eligible": len({r["date"] for r in eligible_bg}),
                "background_g2_events": sum(r["background_g2_count"] for r in eligible_bg),
                "background_days_triggered": len({r["date"] for r in eligible_bg if r["background_g2_count"]}),
                "background_observed_hours": sum(r["background_observed_bins"] for r in eligible_bg) / 3600,
                "background_event_per_observed_hour":
                    (sum(r["background_g2_count"] for r in eligible_bg) /
                     (sum(r["background_observed_bins"] for r in eligible_bg) / 3600)) if eligible_bg else None,
            }
    return records, event_rows, summaries


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_check() -> None:
    sec = np.arange(-2, 6)
    v = np.array([[11.0]] * len(sec))
    valid = np.ones(len(sec), dtype=bool)
    events = g2_events(sec, v, valid, np.array([0.0]), np.array([1.0]), 1.0, "R1")
    assert events == [{"completed_s": 1, "origin_bin_s": -2, "score_at_trigger": 11.0}]
    # A run that started before zero cannot become a new stimulus response.
    assert not [e for e in events if e["origin_bin_s"] >= 0 and 0 < e["completed_s"] <= 300]
    sec = np.array([-1, 0, 1, 2, 4, 5, 6])
    v = np.array([[0.0], [11.0], [11.0], [11.0], [11.0], [11.0], [11.0]])
    events = g2_events(sec, v, np.ones(len(sec), dtype=bool), np.array([0.0]), np.array([1.0]), 1.0, "R1")
    assert [(e["completed_s"], e["origin_bin_s"]) for e in events] == [(3, 0), (7, 4)]
    valid[1] = False
    assert g2_events(np.array([0, 1, 2]), np.array([[11.0]] * 3),
                     np.array([True, False, True]), np.array([0.0]), np.array([1.0]), 1.0, "R1") == []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("Event-boundary self-check passed")
        return
    if OUT.exists():
        raise FileExistsError("Locked run already exists; never overwrite the first result")
    self_check()
    entries, arrays = load()
    fit_info = fit(entries, arrays)
    records, events, summary = evaluate(entries, arrays, fit_info)
    OUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUT / "per_record.csv", records)
    write_csv(OUT / "events.csv", events, ["id", "date", "split", "class", "method", "completed_s", "origin_bin_s", "score_at_trigger", "scope"])
    run = {"source_sha256": SOURCE_SHA, "manifest_sha256": sha256(MANIFEST),
           "code_sha256": sha256(Path(__file__)),
           "protocol": "docs/stage9-a1-uci362-source-audit-and-protocol.md",
           "quantile": 0.999, "aggregation": "8-channel median absolute standardized resistance, 3 consecutive valid seconds",
           "calibration": fit_info, "summary": summary}
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"thresholds": fit_info["thresholds"], "test": summary["test"]}, indent=2))


if __name__ == "__main__":
    main()
