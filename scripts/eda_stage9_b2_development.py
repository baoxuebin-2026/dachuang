#!/usr/bin/env python3
"""Diagnostic-only EDA of the B2 mine source on development dates.

No prediction labels/windows or fitted models are generated; only development
rows enter any EDA statistics, and iteration stops at the first later chunk.
Source data is publisher aligned/forward-filled; unchanged seconds are not
independent sensor measurements. Re-run against the SHA-checked original ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "405a8b27004a2e318bf213cf209398d3d58b45bb3c7a78f159e69003ff459311"
TARGETS = ("MM256", "MM263", "MM264")
DIAGNOSTIC_CHANNELS = ("AN311", "AN422", "AN423", "TP1721", "RH1722", "BA1723", "WM868")
COLUMNS = ("year", "month", "day", "hour", "minute", "second", *TARGETS, *DIAGNOSTIC_CHANNELS)
DEV_START = "2014-03-02"
DEV_END = "2014-04-30"


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def quantiles(a: np.ndarray) -> dict:
    finite = a[np.isfinite(a)]
    if not len(finite):
        return {}
    values = np.quantile(finite, (0, 0.5, 0.9, 0.95, 0.99, 0.999, 1))
    return dict(zip(("min", "p50", "p90", "p95", "p99", "p999", "max"), map(float, values)))


def eda(path: Path) -> dict:
    if source_hash(path) != SOURCE_SHA:
        raise ValueError("Raw ZIP SHA mismatch")
    partitions: dict[str, list[np.ndarray]] = {name: [] for name in TARGETS}
    day_parts: list[np.ndarray] = []
    sampled: dict[str, list[np.ndarray]] = {name: [] for name in (*TARGETS, *DIAGNOSTIC_CHANNELS)}
    daily: dict[str, dict] = defaultdict(lambda: {"rows": 0, "high_seconds": Counter(), "bad_seconds": Counter()})
    total = 0
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["methane_data.csv"] or archive.testzip() is not None:
            raise ValueError("Unexpected archive contents or CRC mismatch")
        with archive.open("methane_data.csv") as stream:
            for frame in pd.read_csv(stream, usecols=COLUMNS, chunksize=150_000):
                days = pd.to_datetime(frame[["year", "month", "day"]]).dt.strftime("%Y-%m-%d")
                if days.iloc[0] > DEV_END:
                    break
                keep = days.between(DEV_START, DEV_END)
                if not keep.any():
                    continue
                frame = frame.loc[keep]
                days = days.loc[keep]
                total += len(frame)
                day_parts.append((frame["month"].to_numpy(dtype=np.int16) * 32
                                  + frame["day"].to_numpy(dtype=np.int16)))
                for d, group in frame.groupby(days, sort=False):
                    daily[d]["rows"] += len(group)
                    for name in TARGETS:
                        a = group[name].to_numpy(dtype=np.float64)
                        daily[d]["high_seconds"][name] += int((a >= 1.0).sum())
                        daily[d]["bad_seconds"][name] += int((~np.isfinite(a) | (a < 0) | (a > 30)).sum())
                for name in TARGETS:
                    partitions[name].append(frame[name].to_numpy(dtype=np.float32))
                minute_end = frame["second"].eq(59)
                for name in sampled:
                    sampled[name].append(frame.loc[minute_end, name].to_numpy(dtype=np.float32))

    arrays = {name: np.concatenate(parts) for name, parts in partitions.items()}
    day_ids = np.concatenate(day_parts)
    minute_arrays = {name: np.concatenate(parts) for name, parts in sampled.items()}
    if total != 5_180_400:
        raise ValueError(f"Unexpected development size: {total}")
    result = {
        "status": "diagnostic_only; development dates exclusively; not a predictive result",
        "source_archive_sha256": SOURCE_SHA,
        "development_dates": [DEV_START, DEV_END],
        "rows": total,
        "minute_samples_for_distribution": int(len(minute_arrays["MM256"])),
        "by_target": {},
        "diagnostic_channels_minute_distribution": {
            name: {"quantiles": quantiles(a), "zero_fraction": float(np.mean(a == 0)),
                   "bad_or_nonfinite": int((~np.isfinite(a)).sum())}
            for name, a in minute_arrays.items() if name in DIAGNOSTIC_CHANNELS
        },
        "daily": {},
    }
    for name, a in arrays.items():
        high = a >= 1.0
        starts, ends = runs(high)
        length = ends - starts
        sustained = length >= 30
        sustained_starts = starts[sustained]
        sustained_ends = ends[sustained]
        preceding_low = np.r_[starts[0], starts[1:] - ends[:-1]] if len(starts) else np.array([], dtype=int)
        # This is an event-feasibility diagnostic, not a count of label windows.
        clean_180 = sustained & (preceding_low >= 180)
        clusters_30min = int(
            len(sustained_starts)
            and (1 + (sustained_starts[1:] - sustained_ends[:-1] >= 1800).sum())
        )
        finite_valid = np.isfinite(a) & (a >= 0) & (a <= 30)
        minute = minute_arrays[name]
        gaps = np.diff(starts[length >= 30])
        # The source audit found exactly one timestamp gap in development.
        # This diagnostic uses row positions for event-spacing statistics only;
        # none of these distances become ground-truth labels.
        result["by_target"][name] = {
            "seconds_below_zero": int((a < 0).sum()),
            "seconds_above_30": int((a > 30).sum()),
            "nonfinite_seconds": int((~np.isfinite(a)).sum()),
            "minute_value_quantiles_pct_ch4": quantiles(minute),
            "minute_fraction_at_or_above_1pct": float(np.mean(minute >= 1)),
            "seconds_same_as_previous_fraction": float(np.mean(a[1:] == a[:-1])),
            "exact_zero_seconds_fraction": float(np.mean(a == 0)),
            "distinct_minute_values": int(len(np.unique(minute))),
            "seconds_at_or_above_2pct": int((a >= 2).sum()),
            "seconds_at_or_above_5pct": int((a >= 5).sum()),
            "seconds_at_or_above_10pct": int((a >= 10).sum()),
            "sustained_runs_with_180s_clean_before_onset": int(clean_180.sum()),
            "sustained_runs_as_30min_separated_clusters_diagnostic_only": clusters_30min,
            "crossing_run_length_seconds": {
                "total": int(len(length)), "one_to_nine": int((length < 10).sum()),
                "ten_to_29": int(((length >= 10) & (length < 30)).sum()),
                "at_least_30": int((length >= 30).sum()),
                "at_least_60": int((length >= 60).sum()),
                "at_least_180": int((length >= 180).sum()),
                "quantiles_all": quantiles(length),
            },
            "start_spacing_ge_30s_runs_in_rows": quantiles(gaps),
            "valid_fraction_0_to_30": float(np.mean(finite_valid)),
        }
        for day_id in np.unique(day_ids[sustained_starts]):
            mask = day_ids[starts] == day_id
            d = f"2014-{int(day_id) // 32:02d}-{int(day_id) % 32:02d}"
            daily[d].setdefault("sustained_run_starts", Counter())[name] = int((sustained & mask).sum())
            daily[d].setdefault("clean_180_sustained_starts", Counter())[name] = int((clean_180 & mask).sum())
    for d, counts in sorted(daily.items()):
        result["daily"][d] = {
            "rows": counts["rows"],
            "high_seconds": dict(counts["high_seconds"]),
            "bad_seconds": dict(counts["bad_seconds"]),
            "sustained_run_starts": dict(counts.get("sustained_run_starts", {})),
            "clean_180_sustained_starts": dict(counts.get("clean_180_sustained_starts", {})),
        }
    result["mm256_days_with_most_high_seconds"] = sorted(
        ((d, int(v["high_seconds"]["MM256"])) for d, v in daily.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )[:10]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage9_b2/development_eda.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = eda(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "targets": report["by_target"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
