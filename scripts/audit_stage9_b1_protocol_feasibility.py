#!/usr/bin/env python3
"""Check DLQ day splits and sampling coverage without reading methane amplitudes.

Only source availability and release-log timing are inspected. This script does
not fit or score a detector. Clone wsdaniels/DLQ at the pinned commit, then use
--source-dir to point at its input_data directory. Do not redistribute raw CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "development": ["2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12"],
    "selection": ["2022-05-13"],
    "locked_evaluation": ["2022-05-14", "2022-05-15"],
}
EXPECTED = {
    "ADED_data_clean_SAMPLE.csv": "fb1fd21e77fef2ad508e2de2697743c1bb010b616457cde29a1480bf79d7c26b",
    "leak_data.csv": "20792a9b8c318ff3490c47f8e071b2dd22c69caeb5ce3356a4a6f6298e048b6a",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit(source_dir: Path) -> dict:
    for name, sha in EXPECTED.items():
        if digest(source_dir / name) != sha:
            raise ValueError(f"Source file differs from pinned audit: {name}")
    # usecols excludes methane, coordinates, and any output or author predictions.
    obs = pd.read_csv(source_dir / "ADED_data_clean_SAMPLE.csv", usecols=["time", "name", "wind.speed"], parse_dates=["time"])
    log = pd.read_csv(source_dir / "leak_data.csv", usecols=["tc_ExpStartDatetime", "tc_ExpEndDatetime", "tc_ExpEPCount"], parse_dates=["tc_ExpStartDatetime", "tc_ExpEndDatetime"])
    grid = pd.date_range(obs.time.min(), obs.time.max(), freq="min", tz="UTC")
    if obs.time.dt.second.ne(0).any() or obs.time.isna().any():
        raise ValueError("Unexpected observation timestamp format")
    starts = log.tc_ExpStartDatetime
    ends = log.tc_ExpEndDatetime
    in_sample = (starts <= grid[-1]) & (ends >= grid[0])
    rel = log.loc[in_sample].sort_values("tc_ExpStartDatetime")
    active = np.zeros(len(grid), dtype=bool)
    negative_washout = np.zeros(len(grid), dtype=bool)
    for row in rel.itertuples():
        active |= (grid >= row.tc_ExpStartDatetime) & (grid <= row.tc_ExpEndDatetime)
        negative_washout |= (grid > row.tc_ExpEndDatetime) & (grid <= row.tc_ExpEndDatetime + pd.Timedelta(minutes=30))
    starts_by_day = Counter(rel.tc_ExpStartDatetime.dt.strftime("%Y-%m-%d"))
    sensor_coverage = obs.groupby("time")["name"].nunique().reindex(grid.tz_localize(None), fill_value=0).to_numpy()
    daily = {}
    for day in sorted(set(grid.strftime("%Y-%m-%d"))):
        mask = np.asarray(grid.strftime("%Y-%m-%d") == day)
        after_warmup = np.asarray(grid >= grid.floor("D") + pd.Timedelta(minutes=60))
        eligible_measurements = mask & after_warmup & (sensor_coverage >= 6)
        on_day = obs.loc[obs.time.dt.strftime("%Y-%m-%d") == day]
        daily[day] = {
            "release_starts": starts_by_day[day],
            "release_minutes": int(np.sum(mask & active)),
            "no_release_minutes": int(np.sum(mask & ~active)),
            "eligible_no_release_minutes_after_30m_washout": int(np.sum(mask & ~active & ~negative_washout)),
            "records": len(on_day),
            "distinct_observation_minutes": int(on_day.time.nunique()),
            "duplicate_sensor_minute_records": int(on_day.duplicated(["name", "time"]).sum()),
            "wind_present_records": int(on_day["wind.speed"].notna().sum()),
            "minutes_with_fewer_than_6_sensors": int(np.sum(mask & (sensor_coverage < 6))),
            "minutes_available_after_60m_warmup_and_6_sensor_gate": int(np.sum(eligible_measurements)),
            "evaluable_no_release_minutes_after_washout_warmup_and_gate": int(np.sum(eligible_measurements & ~active & ~negative_washout)),
        }
    by_split = {
        split: {
            "days": days,
            "release_starts": sum(daily[d]["release_starts"] for d in days),
            "eligible_no_release_minutes": sum(daily[d]["eligible_no_release_minutes_after_30m_washout"] for d in days),
            "evaluable_no_release_minutes_after_washout_warmup_and_gate": sum(
                daily[d]["evaluable_no_release_minutes_after_washout_warmup_and_gate"] for d in days
            ),
        }
        for split, days in SPLITS.items()
    }
    return {
        "source_commit": "9024cc18a9e7dc49c9eff6bc268493ca931e7d0c",
        "source_sha256": EXPECTED,
        "utc_splits_declared_before_daily_count_inspection": SPLITS,
        "release_rows_overlapping_sample": len(rel),
        "overlapping_multisource_log_rows": int((rel.tc_ExpEPCount != 1).sum()),
        "duplicate_sensor_minute_records": int(obs.duplicated(["name", "time"]).sum()),
        "wind_present_by_sensor": {str(k): int(v) for k, v in obs.groupby("name")["wind.speed"].apply(lambda x: x.notna().sum()).items()},
        "daily": daily,
        "by_split": by_split,
        "scope": "Read-only release-timing and measurement-availability audit; no methane readings, trained model, predictions, or scores were read.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/stage9_b1/protocol_feasibility.json")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    out = audit(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["by_split"], indent=2))


if __name__ == "__main__":
    main()
