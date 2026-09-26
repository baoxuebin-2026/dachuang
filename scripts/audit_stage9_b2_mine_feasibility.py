#!/usr/bin/env python3
"""Read-only source audit for a *proposed*, not yet frozen, mine forecast task.

Counts daily threshold runs and data quality. It does not build prediction
windows, fit a model, or calculate an alert score. Original ZIP stays ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "405a8b27004a2e318bf213cf209398d3d58b45bb3c7a78f159e69003ff459311"
TARGETS = ("MM263", "MM264", "MM256")
SPLITS = {
    "development": ("2014-03-02", "2014-04-30"),
    "selection": ("2014-05-01", "2014-05-23"),
    "reserved": ("2014-05-24", "2014-06-16"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit(archive: Path) -> dict:
    if digest(archive) != EXPECTED_SHA:
        raise ValueError("Original ZIP SHA-256 differs from publisher value")
    per_day = defaultdict(Counter)
    per_target = {name: Counter() for name in TARGETS}
    run_len = {name: 0 for name in TARGETS}
    run_day = {name: None for name in TARGETS}
    last_high_second = {name: None for name in TARGETS}
    prior_stamp = None
    previous_day = None
    day_ordinal = None
    n = 0

    def finish(name: str) -> None:
        length = run_len[name]
        if length:
            per_target[name]["raw_crossing_runs"] += 1
            per_day[run_day[name]][f"{name}_raw_runs"] += 1
            for threshold in (10, 30, 60):
                if length >= threshold:
                    per_target[name][f"runs_ge_{threshold}s"] += 1
                    per_day[run_day[name]][f"{name}_runs_ge_{threshold}s"] += 1
        run_len[name] = 0

    with zipfile.ZipFile(archive) as z:
        if z.namelist() != ["methane_data.csv"] or z.testzip() is not None:
            raise ValueError("Unexpected contents or CRC failure")
        with z.open("methane_data.csv") as stream:
            reader = csv.reader((line.decode("utf-8-sig") for line in stream))
            header = next(reader)
            indexes = {name: header.index(name) for name in TARGETS}
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Unexpected width at row {n + 2}")
                d = tuple(int(v) for v in row[:3])
                if d != previous_day:
                    day_ordinal = date(*d).toordinal()
                    previous_day = d
                stamp = day_ordinal * 86400 + int(row[3]) * 3600 + int(row[4]) * 60 + int(row[5])
                day = date(*d).isoformat()
                consecutive = prior_stamp is None or stamp == prior_stamp + 1
                if not consecutive:
                    per_day[day]["timestamp_discontinuity_rows"] += 1
                    for name in TARGETS:
                        finish(name)
                        last_high_second[name] = None
                per_day[day]["rows"] += 1
                n += 1
                for name, index in indexes.items():
                    value = float(row[index])
                    if value < 0:
                        per_day[day][f"{name}_negative"] += 1
                        per_target[name]["negative_rows"] += 1
                    if value > 30:
                        per_day[day][f"{name}_above_30"] += 1
                        per_target[name]["above_30_rows"] += 1
                    crossing = value >= 1.0
                    if crossing:
                        per_target[name]["crossing_rows"] += 1
                        per_day[day][f"{name}_crossing_rows"] += 1
                        if run_len[name] == 0:
                            run_day[name] = day
                        run_len[name] += 1
                        last_high_second[name] = stamp
                    else:
                        finish(name)
                        if value >= 0 and (last_high_second[name] is None or stamp - last_high_second[name] > 180):
                            per_day[day][f"{name}_current_below_and_past_180s_clear"] += 1
                prior_stamp = stamp
    for name in TARGETS:
        finish(name)
    dates = sorted(per_day)
    by_split = {}
    for split, (lo, hi) in SPLITS.items():
        subset = [d for d in dates if lo <= d <= hi]
        by_split[split] = {
            "declared_dates": [lo, hi], "observed_dates": len(subset),
            "rows": sum(per_day[d]["rows"] for d in subset),
            "targets": {name: {
                metric: sum(per_day[d][f"{name}_{metric}"] for d in subset)
                for metric in ("crossing_rows", "raw_runs", "runs_ge_10s", "runs_ge_30s", "runs_ge_60s",
                               "negative", "above_30", "current_below_and_past_180s_clear")
            } for name in TARGETS},
        }
    return {
        "source": "Mendeley Data 10.17632/yd7vw4c5mk.1",
        "archive_sha256": EXPECTED_SHA,
        "threshold_pct_ch4_for_source_feasibility_only": 1.0,
        "total_rows": n,
        "observed_dates": len(dates),
        "date_splits_chosen_before_daily_events_were_read": SPLITS,
        "by_split": by_split,
        "per_target": {name: dict(c) for name, c in per_target.items()},
        "per_day": {d: dict(per_day[d]) for d in dates},
        "scope": "Raw measured threshold runs, not independent release or accident events; no prediction windows or scores computed. Negative and >30 values counted, not automatically invalidated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage9_b2/source_feasibility.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["by_split"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
