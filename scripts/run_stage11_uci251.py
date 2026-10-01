#!/usr/bin/env python3
"""Frozen UCI 251 methane-only evidence check with position and wind holdouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NAME = re.compile(
    r"^WTD_upload/Methane_1000/(L[1-6])/(20\d{10})_board_setPoint_500V_"
    r"fan_setPoint_(000|060|100)_mfc_setPoint_Methane_1000ppm_p\d+$"
)
QUANTILES = (0.95, 0.99, 0.995, 0.999)
BOARD_COLS = tuple(range(48, 56))  # board5 after four blocks of sentinel + eight channels
READ_COLS = [0, 3, 4, 5, 47, *BOARD_COLS]


@dataclass
class Trial:
    path: str
    location: str
    date: str
    fan: str
    raw_sha256: str
    n_rows: int
    dropped_invalid_sensor_rows: int
    valid: np.ndarray
    values: np.ndarray
    first_mfc_on_s: float | None
    first_mfc_off_s: float | None
    calibration_seconds: int


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(archive: Path) -> tuple[list[Trial], dict]:
    trials = []
    with zipfile.ZipFile(archive) as z:
        members = [item for item in z.infolist() if NAME.fullmatch(item.filename)]
        total = len([x for x in z.infolist() if not x.is_dir()])
        if len(members) != 360:
            raise ValueError(f"Expected 360 methane-heater500 trials, found {len(members)}")
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Duplicate archive paths")
        for member in sorted(members, key=lambda x: x.filename):
            location, timestamp, fan = NAME.fullmatch(member.filename).groups()
            original = z.read(member)  # ZipFile verifies CRC as each member is read.
            frame = pd.read_csv(io.BytesIO(original), sep="\t", header=None,
                                usecols=READ_COLS, dtype=np.float64)
            if len(frame.columns) != len(READ_COLS) or len(frame) < 1000:
                raise ValueError(f"Missing columns or too few records: {member.filename}")
            if not np.all(frame[47].to_numpy() == 1):
                raise ValueError(f"board5 separator is not 1: {member.filename}")
            time_ms = frame[0].to_numpy()
            if not np.isfinite(time_ms).all() or (np.diff(time_ms) < 0).any():
                raise ValueError(f"Corrupt time in {member.filename}")
            seconds = np.floor(time_ms / 1000).astype(int)
            if seconds.min() != 0 or seconds.max() < 259 or seconds.max() > 261:
                raise ValueError(f"Unexpected recording duration: {member.filename}")
            raw = frame[list(BOARD_COLS)].to_numpy()
            complete = np.isfinite(raw).all(axis=1) & (raw > 0).all(axis=1) & (raw < 4096).all(axis=1)
            # A corrupt sensor value invalidates its entire sample, never the
            # independent trial; second validity is still gated on >=50 samples.
            grouped = frame.loc[complete].groupby(seconds[complete], sort=True)[list(BOARD_COLS)]
            counts = grouped.size().reindex(range(260), fill_value=0).to_numpy()
            medians = grouped.median().reindex(range(260)).to_numpy()
            valid = (counts >= 50) & np.isfinite(medians).all(axis=1)
            mfc_active = (frame[[3, 4, 5]].to_numpy() != 0).any(axis=1)
            starts = np.flatnonzero(mfc_active & np.r_[True, ~mfc_active[:-1]])
            ends = np.flatnonzero(~mfc_active & np.r_[False, mfc_active[:-1]])
            trials.append(Trial(
                member.filename, location, timestamp[:8], fan,
                hashlib.sha256(original).hexdigest(), len(frame), int((~complete).sum()), valid, medians,
                float(time_ms[starts[0]] / 1000) if len(starts) else None,
                float(time_ms[ends[0]] / 1000) if len(ends) else None,
                int(valid[5:15].sum()),
            ))
    return trials, {"all_archive_file_count": total, "selected_file_count": len(members)}


def split(t: Trial, experiment: str) -> str:
    if experiment == "position":
        return "train" if t.location in {"L1", "L2", "L3", "L4"} else (
            "validation" if t.location == "L5" else "test")
    if experiment == "wind":
        return {"000": "train", "060": "validation", "100": "test"}[t.fan]
    raise ValueError(experiment)


def score(t: Trial, scale: np.ndarray) -> np.ndarray:
    out = np.full(260, np.nan)
    if t.calibration_seconds == 0:
        return out
    center = np.median(t.values[5:15][t.valid[5:15]], axis=0)
    out[t.valid] = np.median(np.abs((t.values[t.valid] - center) / scale), axis=1)
    return out


def fit(trials: list[Trial], experiment: str) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    training = [t for t in trials if split(t, experiment) == "train" and t.calibration_seconds]
    if not training:
        raise ValueError("No usable training files")
    centers = np.array([np.median(t.values[5:15][t.valid[5:15]], axis=0) for t in training])
    global_center = np.median(centers, axis=0)
    backgrounds = np.concatenate([
        (t.values[15:20][t.valid[15:20]] - center)
        for t, center in zip(training, centers)
    ])
    if len(backgrounds) == 0:
        raise ValueError("No valid training clean-air seconds")
    med = np.median(backgrounds, axis=0)
    mad = 1.4826 * np.median(np.abs(backgrounds - med), axis=0)
    scale = np.maximum(mad, np.maximum(0.001 * np.abs(global_center), 0.001))
    scores = {t.path: score(t, scale) for t in trials}
    return scale, scores, len(backgrounds)


def events(series: np.ndarray, threshold: float) -> dict:
    if not (threshold >= 0 and np.isfinite(threshold)):
        raise ValueError(f"Invalid threshold {threshold}")
    consecutive = 0
    streak_start = None
    pre = 0
    post = []
    seeded = 0
    previous_g2 = False
    for second in range(15, 200):
        hit = bool(np.isfinite(series[second]) and series[second] > threshold)
        if hit:
            if consecutive == 0:
                streak_start = second + 1
            consecutive += 1
        else:
            consecutive = 0
            streak_start = None
        g2 = consecutive >= 3
        when = second + 1
        if g2 and not previous_g2:
            if when <= 20:
                pre += 1
            elif streak_start is not None and streak_start <= 20:
                seeded += 1
            else:
                post.append(when)
        previous_g2 = g2
    first = post[0] if post else None
    return {"pre_release_trigger": int(pre > 0), "pre_release_events": pre,
            "pre_release_seeded_events": seeded,
            "during_release_detected": int(first is not None),
            "within_60s": int(first is not None and first <= 80),
            "first_observed_s": first,
            "observed_delay_s": first - 20 if first is not None else None}


def summarize(rows: list[dict]) -> dict:
    delays = [r["observed_delay_s"] for r in rows if r["observed_delay_s"] is not None]
    return {"n_files": len(rows), "valid_calibration_files": sum(r["calibration_seconds"] > 0 for r in rows),
            "pre_release_trigger_files": sum(r["pre_release_trigger"] for r in rows),
            "within_60s_files": sum(r["within_60s"] for r in rows),
            "during_release_detected_files": sum(r["during_release_detected"] for r in rows),
            "detected_median_observed_delay_s": float(np.median(delays)) if delays else None,
            "median_valid_monitor_seconds": float(np.median([r["valid_monitor_seconds"] for r in rows])) if rows else None}


def candidate(trials: list[Trial], experiment: str, scores: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    training_seconds = np.concatenate([
        scores[t.path][15:20][np.isfinite(scores[t.path][15:20])]
        for t in trials if split(t, experiment) == "train"
    ])
    if len(training_seconds) == 0:
        raise ValueError("No train background scores")
    records = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(training_seconds, quantile))
        rows = [events(scores[t.path], threshold) for t in trials
                if split(t, experiment) == "validation"]
        stats = summarize([{**r, "calibration_seconds": 1,
                            "valid_monitor_seconds": 0} for r in rows])
        records.append({"quantile": quantile, "threshold": threshold,
                        "validation": stats})
    def ranking(record: dict) -> tuple:
        val = record["validation"]
        delay = val["detected_median_observed_delay_s"]
        return (-val["pre_release_trigger_files"], val["within_60s_files"],
                -(delay if delay is not None else 999999), record["quantile"])
    return max(records, key=ranking), records


def rows_for(trials: list[Trial], experiment: str, scores: dict[str, np.ndarray], threshold: float) -> list[dict]:
    rows = []
    for trial in trials:
        e = events(scores[trial.path], threshold)
        rows.append({"experiment": experiment, "file": trial.path, "date": trial.date,
                     "position": trial.location, "wind_setpoint": trial.fan,
                     "split": split(trial, experiment),
                     "source_sha256": trial.raw_sha256,
                     "source_rows": trial.n_rows,
                     "dropped_invalid_sensor_rows": trial.dropped_invalid_sensor_rows,
                     "first_mfc_on_s_audit_only": trial.first_mfc_on_s,
                     "first_mfc_off_s_audit_only": trial.first_mfc_off_s,
                     "calibration_seconds": trial.calibration_seconds,
                     "valid_background_seconds": int(trial.valid[15:20].sum()),
                     "valid_monitor_seconds": int(trial.valid[15:200].sum()),
                     **e})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trials, inventory = load(args.archive)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"run_id": "stage11-uci251-methane-20261001-v1",
              "archive_sha256": hash_file(args.archive),
              "script_sha256": hash_file(Path(__file__)),
              "protocol_git_head_before_results": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "runtime": {"python": platform.python_version(),
                          "numpy": np.__version__, "pandas": pd.__version__},
              "inventory": inventory, "analyses": {}}
    all_rows = []
    for experiment in ("position", "wind"):
        scale, scores, train_background_seconds = fit(trials, experiment)
        choice, candidates = candidate(trials, experiment, scores)
        rows = rows_for(trials, experiment, scores, choice["threshold"])
        all_rows.extend(rows)
        test = [r for r in rows if r["split"] == "test"]
        controls = [events(np.zeros_like(scores[t.path]), choice["threshold"])
                    for t in trials if split(t, experiment) == "test"]
        if any(r["during_release_detected"] or r["pre_release_trigger"] for r in controls):
            raise ValueError("Frozen gas signal still triggered: possible time shortcut")
        by_condition = {}
        key = "wind_setpoint" if experiment == "position" else "position"
        for condition in sorted({r[key] for r in test}):
            by_condition[condition] = summarize([r for r in test if r[key] == condition])
        train_dates = {t.date for t in trials if split(t, experiment) == "train"}
        test_dates = {t.date for t in trials if split(t, experiment) == "test"}
        overlaps_by_pos = {loc: len(
            {t.date for t in trials if t.location == loc and split(t, experiment) == "train"} &
            {t.date for t in trials if t.location == loc and split(t, experiment) == "test"}
        ) for loc in sorted({t.location for t in trials})}
        report["analyses"][experiment] = {
            "split_counts": dict(Counter(split(t, experiment) for t in trials)),
            "train_background_seconds": train_background_seconds,
            "scales": [float(x) for x in scale],
            "candidates": candidates,
            "selected_quantile": choice["quantile"],
            "selected_threshold": choice["threshold"],
            "test": summarize(test),
            "test_by_condition": by_condition,
            "training_test_date_overlap_days": len(train_dates & test_dates),
            "test_unique_days": len(test_dates),
            "training_test_date_overlap_by_position": overlaps_by_pos,
            "frozen_gas_negative_control_test_triggers": 0,
        }
    fieldnames = list(all_rows[0])
    with (args.out / "file_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({e: {"test": a["test"], "selected_quantile": a["selected_quantile"],
                          "date_overlap": a["training_test_date_overlap_days"]}
                      for e, a in report["analyses"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
