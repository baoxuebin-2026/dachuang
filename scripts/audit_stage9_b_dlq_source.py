#!/usr/bin/env python3
"""Read-only audit of the public DLQ sample and *independent* release log.

No sensor-response scores, thresholds, or model fitting are produced.
Clone https://github.com/wsdaniels/DLQ at commit 9024cc18a9e7dc49c9eff6bc268493ca931e7d0c
and pass its input_data directory as --source-dir. Do not redistribute the raw sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "9024cc18a9e7dc49c9eff6bc268493ca931e7d0c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(source_dir: Path) -> dict:
    observations = source_dir / "ADED_data_clean_SAMPLE.csv"
    release_log = source_dir / "leak_data.csv"
    obs = pd.read_csv(observations, parse_dates=["time"])
    logs = pd.read_csv(release_log, parse_dates=["tc_ExpStartDatetime", "tc_ExpEndDatetime"])
    if obs.time.isna().any() or logs[["tc_ExpStartDatetime", "tc_ExpEndDatetime"]].isna().any().any():
        raise ValueError("Unparseable timestamps")
    start, end = obs.time.min(), obs.time.max()
    minutes = pd.date_range(start, end, freq="min", tz="UTC")
    active = np.zeros(len(minutes), dtype=bool)
    overlapping_rows = 0
    for row in logs.itertuples():
        release = (minutes >= row.tc_ExpStartDatetime) & (minutes <= row.tc_ExpEndDatetime)
        if release.any():
            overlapping_rows += 1
            active |= release
    starts = int(np.count_nonzero(active & ~np.r_[False, active[:-1]]))
    # These are prospective labels derived only from the independent release log;
    # they are not safety-danger labels or methane-threshold crossings.
    return {
        "source": "https://github.com/wsdaniels/DLQ",
        "source_commit": SOURCE_COMMIT,
        "observation_file": observations.name,
        "observation_bytes": observations.stat().st_size,
        "observation_sha256": sha256(observations),
        "release_file": release_log.name,
        "release_bytes": release_log.stat().st_size,
        "release_sha256": sha256(release_log),
        "observation_rows": len(obs),
        "observation_columns": list(obs.columns),
        "sensor_names": sorted(obs.name.unique().tolist()),
        "observations_by_sensor": {str(k): int(v) for k, v in obs.name.value_counts().sort_index().items()},
        "observation_start_utc": start.isoformat() + "+00:00",
        "observation_end_utc": end.isoformat() + "+00:00",
        "observation_days": int(obs.time.dt.date.nunique()),
        "missing_methane_rows": int(obs.methane.isna().sum()),
        "missing_wind_speed_rows": int(obs["wind.speed"].isna().sum()),
        "all_release_log_rows": len(logs),
        "release_log_rows_overlapping_sample": overlapping_rows,
        "minute_grid": len(minutes),
        "release_active_minutes": int(active.sum()),
        "no_release_minutes": int((~active).sum()),
        "release_start_transitions": starts,
        "longest_contiguous_no_release_minutes": int(max(
            np.diff(np.r_[-1, np.flatnonzero(active), len(active)]) - 1
        )),
        "audit_scope": "Source fields and independent release coverage only; no model, response accuracy, or safety risk threshold.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage9_b_sources/dlq_source_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Audit would overwrite existing file: {args.output}")
    result = audit(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
