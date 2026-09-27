"""Run the frozen METEC P1 leave-one-date-out threshold diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np


UTC = timezone.utc
FIXED_MST = timezone(timedelta(hours=-7), name="MST")
AMERICA_DENVER = ZoneInfo("America/Denver")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_release_time(value: str) -> datetime:
    return datetime.strptime(value, "%m/%d/%Y %H:%M").replace(tzinfo=UTC)


def parse_local_time(value: str) -> datetime:
    return datetime.strptime(value, "%m/%d/%Y %H:%M:%S.%f")


def local_to_utc(local: datetime, basis: str) -> datetime:
    zone = FIXED_MST if basis == "fixed_mst" else AMERICA_DENVER
    return local.replace(tzinfo=zone).astimezone(UTC)


def merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    merged: list[list[datetime]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def rank_auc(positives: list[float], negatives: list[float]) -> float:
    if not positives or not negatives:
        return math.nan
    combined = sorted(
        [(value, 1) for value in positives] + [(value, 0) for value in negatives]
    )
    positive_rank_sum = 0.0
    index = 0
    while index < len(combined):
        end = index + 1
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            label for _, label in combined[index:end]
        )
        index = end
    n_positive = len(positives)
    n_negative = len(negatives)
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def read_release_intervals(path: Path) -> list[tuple[datetime, datetime]]:
    intervals = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            start = parse_release_time(row["UTCStart"])
            end = parse_release_time(row["UTCEnd"])
            if end <= start:
                hours, minutes, seconds = (
                    int(part) for part in row.get("Duration_HMS", "0:00:00").split(":")
                )
                end = start + timedelta(
                    hours=hours, minutes=minutes, seconds=max(seconds, 1)
                )
            intervals.append((start, end))
    return merge_intervals(intervals)


def read_second_medians(path: Path) -> dict[datetime, float]:
    per_second: dict[datetime, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            local = parse_local_time(row["Time"]).replace(microsecond=0)
            value = float(row["x_CH4_d_ppm"])
            if math.isfinite(value):
                per_second[local].append(value)
    return {second: median(values) for second, values in per_second.items()}


def classify_by_date(
    second_medians: dict[datetime, float],
    intervals: list[tuple[datetime, datetime]],
    basis: str,
) -> dict[str, dict[str, list[float]]]:
    starts = [start.timestamp() for start, _ in intervals]
    ends = [end.timestamp() for _, end in intervals]
    per_date: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"release": [], "outside_release": []}
    )
    for local, value in second_medians.items():
        second = local_to_utc(local, basis).timestamp()
        index = bisect_right(starts, second) - 1
        state = "release" if index >= 0 and second < ends[index] else "outside_release"
        per_date[local.date().isoformat()][state].append(value)
    return dict(per_date)


def leave_one_date_out(
    per_date: dict[str, dict[str, list[float]]],
    min_release: int,
    min_outside: int,
) -> list[dict[str, float | int | str]]:
    eligible = [
        day
        for day, values in sorted(per_date.items())
        if len(values["release"]) >= min_release
        and len(values["outside_release"]) >= min_outside
    ]
    rows = []
    for held_out in eligible:
        training_outside = [
            value
            for day, values in per_date.items()
            if day != held_out
            for value in values["outside_release"]
        ]
        if not training_outside:
            raise ValueError(f"no training outside-release observations for {held_out}")
        test_release = per_date[held_out]["release"]
        test_outside = per_date[held_out]["outside_release"]
        p95, p99 = np.quantile(training_outside, [0.95, 0.99], method="linear")
        rows.append(
            {
                "held_out_date": held_out,
                "training_outside_seconds": len(training_outside),
                "release_seconds": len(test_release),
                "outside_release_seconds": len(test_outside),
                "p95_threshold_ppm": float(p95),
                "p95_release_fraction": float(np.mean(np.asarray(test_release) > p95)),
                "p95_outside_fraction": float(np.mean(np.asarray(test_outside) > p95)),
                "p99_threshold_ppm": float(p99),
                "p99_release_fraction": float(np.mean(np.asarray(test_release) > p99)),
                "p99_outside_fraction": float(np.mean(np.asarray(test_outside) > p99)),
                "rank_auc": rank_auc(test_release, test_outside),
            }
        )
    return rows


def bootstrap_median_interval(
    values: list[float], seed: int, replicates: int
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    medians = np.median(array[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975], method="linear")
    return {"low": float(low), "high": float(high)}


def summarize_rows(
    rows: list[dict[str, float | int | str]], seed: int, replicates: int
) -> dict[str, object]:
    metrics = [
        "p95_release_fraction",
        "p95_outside_fraction",
        "p99_release_fraction",
        "p99_outside_fraction",
        "rank_auc",
    ]
    summary: dict[str, object] = {"eligible_dates": len(rows), "metrics": {}}
    for offset, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        summary["metrics"][metric] = {
            "macro_median": float(q50),
            "macro_q25": float(q25),
            "macro_q75": float(q75),
            "date_cluster_bootstrap_median_95_interval": bootstrap_median_interval(
                values, seed + offset, replicates
            ),
        }
    for quantile in ("p95", "p99"):
        thresholds = [float(row[f"{quantile}_threshold_ppm"]) for row in rows]
        summary[f"{quantile}_threshold_ppm"] = {
            "min": min(thresholds),
            "median": median(thresholds),
            "max": max(thresholds),
        }
    return summary


def write_rows(path: Path, basis: str, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_basis",
        "held_out_date",
        "training_outside_seconds",
        "release_seconds",
        "outside_release_seconds",
        "p95_threshold_ppm",
        "p95_release_fraction",
        "p95_outside_fraction",
        "p99_threshold_ppm",
        "p99_release_fraction",
        "p99_outside_fraction",
        "rank_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"time_basis": basis, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--releases", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "approved_P1_frozen_before_run":
        raise ValueError("P1 config is not approved and frozen")
    expected = config["source"]
    actual_release_hash = sha256(args.releases)
    actual_observation_hash = sha256(args.observations)
    if actual_release_hash != expected["releases_file"]["sha256"]:
        raise ValueError("release file hash does not match frozen config")
    if actual_observation_hash != expected["observations_file"]["sha256"]:
        raise ValueError("observation file hash does not match frozen config")

    intervals = read_release_intervals(args.releases)
    second_medians = read_second_medians(args.observations)
    min_release = config["analysis_unit"]["eligible_day_min_release_seconds"]
    min_outside = config["analysis_unit"]["eligible_day_min_outside_release_seconds"]
    seed = config["metrics"]["bootstrap"]["seed"]
    replicates = config["metrics"]["bootstrap"]["replicates"]
    results = {}
    for basis in ("fixed_mst", "america_denver"):
        per_date = classify_by_date(second_medians, intervals, basis)
        rows = leave_one_date_out(per_date, min_release, min_outside)
        write_rows(args.out / f"per_date_{basis}.csv", basis, rows)
        results[basis] = {
            "all_dates_with_observations": len(per_date),
            **summarize_rows(rows, seed, replicates),
        }

    summary = {
        "status": "completed_auxiliary_date_grouped_diagnostic",
        "claim_scope": (
            "Controlled-release state versus sparse single-point methane observations; "
            "not continuous event detection, warning lead time, or coking-park validation."
        ),
        "frozen_config_sha256": sha256(args.config),
        "source_sha256": {
            "releases": actual_release_hash,
            "observations": actual_observation_hash,
        },
        "second_level_observations": len(second_medians),
        "merged_site_release_intervals": len(intervals),
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run = {
        "run_id": "stage9-metec-p1-20260927-v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "script_sha256": sha256(Path(__file__)),
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "command": " ".join(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "random_seed": seed,
        "bootstrap_replicates": replicates,
        "output_path": str(args.out),
        "status": "completed_auxiliary_only",
    }
    (args.out / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
