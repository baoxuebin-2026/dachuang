"""Read-only audit of the METEC release log and 8 Hz methane observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo


UTC = timezone.utc
MOUNTAIN = ZoneInfo("America/Denver")
FIXED_MST = timezone(timedelta(hours=-7), name="MST")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_release_time(value: str) -> datetime:
    return datetime.strptime(value, "%m/%d/%Y %H:%M").replace(tzinfo=UTC)


def parse_observation_time(value: str, time_basis: str) -> datetime:
    zone = FIXED_MST if time_basis == "fixed_mst" else MOUNTAIN
    local = datetime.strptime(value, "%m/%d/%Y %H:%M:%S.%f").replace(tzinfo=zone)
    return local.astimezone(UTC)


def merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[list[datetime]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            key: None
            for key in ("min", "p05", "p25", "median", "p75", "p95", "p99", "max")
        }
    ordered = sorted(values)

    def q(probability: float) -> float:
        index = probability * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    return {
        "min": ordered[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": median(ordered),
        "p75": q(0.75),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
    }


def covered_seconds(seconds: set[int], start: int, end: int) -> int:
    return sum(1 for value in seconds if start <= value < end)


def segment_stats(seconds: list[int], max_gap_seconds: int = 2) -> dict[str, int]:
    if not seconds:
        return {"segments": 0, "longest_observed_seconds": 0, "longest_span_seconds": 0}
    segment_count = 1
    segment_start = seconds[0]
    observed_count = 1
    longest_observed = 1
    longest_span = 1
    for previous, current in zip(seconds, seconds[1:]):
        if current - previous > max_gap_seconds:
            segment_count += 1
            segment_start = current
            observed_count = 1
        else:
            observed_count += 1
        longest_observed = max(longest_observed, observed_count)
        longest_span = max(longest_span, current - segment_start + 1)
    return {
        "segments": segment_count,
        "longest_observed_seconds": longest_observed,
        "longest_span_seconds": longest_span,
    }


def rank_auc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    combined = sorted([(value, 1) for value in positives] + [(value, 0) for value in negatives])
    positive_rank_sum = 0.0
    index = 0
    while index < len(combined):
        end = index + 1
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in combined[index:end])
        index = end
    n_positive = len(positives)
    n_negative = len(negatives)
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument(
        "--observation-time-basis",
        choices=("fixed_mst", "america_denver"),
        default="fixed_mst",
        help=(
            "Interpret the observation Time column as fixed UTC-7 (default) or as the "
            "America/Denver civil-time zone with daylight-saving transitions."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    release_rows = []
    release_headers: list[str] = []
    with args.releases.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        release_headers = list(reader.fieldnames or [])
        for row in reader:
            start = parse_release_time(row["UTCStart"])
            end = parse_release_time(row["UTCEnd"])
            if end <= start:
                duration_text = row.get("Duration_HMS", "0:00:00")
                hours, minutes, seconds = (int(part) for part in duration_text.split(":"))
                end = start + timedelta(hours=hours, minutes=minutes, seconds=max(seconds, 1))
            release_rows.append(
                {
                    "event_id": row["EventID"],
                    "point": row["EmissionPoint"],
                    "start": start,
                    "end": end,
                    "rate_kg_h": float(row["Actual_Emission_kg_h"]),
                }
            )

    release_intervals = [(row["start"], row["end"]) for row in release_rows]
    episodes = merge_intervals(release_intervals)
    starts = [start.timestamp() for start, _ in episodes]
    ends = [end.timestamp() for _, end in episodes]

    observation_headers: list[str] = []
    observation_rows = 0
    invalid_rows = 0
    missing_by_column: Counter[str] = Counter()
    frequency_counts: Counter[str] = Counter()
    first_time: datetime | None = None
    last_time: datetime | None = None
    previous_time: datetime | None = None
    backward_timestamps = 0
    duplicate_timestamps = 0
    gaps_over_2s = 0
    largest_gap_seconds = 0.0
    per_second: dict[int, list[float]] = defaultdict(list)

    with args.observations.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        observation_headers = list(reader.fieldnames or [])
        for row in reader:
            observation_rows += 1
            for key, value in row.items():
                if value is None or value.strip() == "":
                    missing_by_column[key] += 1
            try:
                timestamp = parse_observation_time(row["Time"], args.observation_time_basis)
                concentration = float(row["x_CH4_d_ppm"])
                if not math.isfinite(concentration):
                    raise ValueError("non-finite concentration")
            except (ValueError, TypeError):
                invalid_rows += 1
                continue
            frequency_counts[row.get("Frequency_CH4", "")] += 1
            if first_time is None:
                first_time = timestamp
            last_time = timestamp
            if previous_time is not None:
                delta = (timestamp - previous_time).total_seconds()
                if delta < 0:
                    backward_timestamps += 1
                elif delta == 0:
                    duplicate_timestamps += 1
                elif delta > 2:
                    gaps_over_2s += 1
                    largest_gap_seconds = max(largest_gap_seconds, delta)
            previous_time = timestamp
            per_second[int(timestamp.timestamp())].append(concentration)

    observed_seconds = set(per_second)
    ordered_observed_seconds = sorted(observed_seconds)
    during_values: list[float] = []
    background_values: list[float] = []
    during_seconds: list[int] = []
    background_seconds: list[int] = []
    daily_counts: dict[str, Counter[str]] = defaultdict(Counter)
    daily_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"release": [], "outside_release": []}
    )
    for second, values in per_second.items():
        episode_index = bisect_right(starts, second) - 1
        is_release = episode_index >= 0 and second < ends[episode_index]
        value = median(values)
        if is_release:
            during_values.append(value)
            during_seconds.append(second)
            state = "release"
        else:
            background_values.append(value)
            background_seconds.append(second)
            state = "outside_release"
        local_zone = FIXED_MST if args.observation_time_basis == "fixed_mst" else MOUNTAIN
        local_day = datetime.fromtimestamp(second, UTC).astimezone(local_zone).date().isoformat()
        daily_counts[local_day][state] += 1
        daily_values[local_day][state].append(value)

    episode_diagnostics = []
    for index, (start, end) in enumerate(episodes, start=1):
        start_s = int(start.timestamp())
        end_s = int(end.timestamp())
        pre = covered_seconds(observed_seconds, start_s - 600, start_s)
        during = covered_seconds(observed_seconds, start_s, end_s)
        first_five = covered_seconds(observed_seconds, start_s, min(end_s, start_s + 300))
        post = covered_seconds(observed_seconds, end_s, end_s + 600)
        episode_diagnostics.append(
            {
                "episode": index,
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "duration_seconds": int((end - start).total_seconds()),
                "pre_10min_observed_seconds": pre,
                "first_5min_observed_seconds": first_five,
                "during_observed_seconds": during,
                "post_10min_observed_seconds": post,
            }
        )

    strict_episodes = [
        item
        for item in episode_diagnostics
        if item["pre_10min_observed_seconds"] >= 480
        and item["first_5min_observed_seconds"] >= min(240, item["duration_seconds"])
    ]
    complete_context_episodes = [
        item
        for item in strict_episodes
        if item["post_10min_observed_seconds"] >= 480
    ]

    background_quantiles = quantiles(background_values)
    pooled_auc = rank_auc(during_values, background_values)
    daily_auc = {
        day: rank_auc(values["release"], values["outside_release"])
        for day, values in sorted(daily_values.items())
        if values["release"] and values["outside_release"]
    }
    finite_daily_auc = [value for value in daily_auc.values() if value is not None]
    exploratory_thresholds = {}
    for label in ("p95", "p99"):
        threshold = background_quantiles[label]
        if threshold is None:
            continue
        exploratory_thresholds[label] = {
            "threshold_ppm": threshold,
            "release_seconds_above_fraction": sum(value > threshold for value in during_values)
            / len(during_values),
            "outside_release_seconds_above_fraction": sum(
                value > threshold for value in background_values
            )
            / len(background_values),
        }

    concurrent_changes = []
    for row in release_rows:
        concurrent_changes.append((row["start"], 1))
        concurrent_changes.append((row["end"], -1))
    concurrency = 0
    max_concurrency = 0
    for _, change in sorted(concurrent_changes, key=lambda item: (item[0], item[1])):
        concurrency += change
        max_concurrency = max(max_concurrency, concurrency)

    result = {
        "source_files": {
            "releases": {
                "name": args.releases.name,
                "bytes": args.releases.stat().st_size,
                "sha256": sha256(args.releases),
            },
            "observations": {
                "name": args.observations.name,
                "bytes": args.observations.stat().st_size,
                "sha256": sha256(args.observations),
            },
        },
        "time_interpretation": {
            "release_columns": "UTC, as documented by publisher",
            "observation_column": (
                "fixed UTC-7 (MST; no daylight-saving shift)"
                if args.observation_time_basis == "fixed_mst"
                else "America/Denver local time converted with daylight-saving rules"
            ),
            "observation_time_basis_argument": args.observation_time_basis,
        },
        "release_log": {
            "headers": release_headers,
            "rows": len(release_rows),
            "unique_event_ids": len({row["event_id"] for row in release_rows}),
            "unique_release_points": len({row["point"] for row in release_rows}),
            "site_level_merged_episodes": len(episodes),
            "max_simultaneous_release_points": max_concurrency,
            "first_start_utc": min(row["start"] for row in release_rows).isoformat(),
            "last_end_utc": max(row["end"] for row in release_rows).isoformat(),
        },
        "observations": {
            "headers": observation_headers,
            "rows": observation_rows,
            "valid_rows": observation_rows - invalid_rows,
            "invalid_rows": invalid_rows,
            "missing_by_column": dict(missing_by_column),
            "frequency_counts": dict(frequency_counts),
            "first_utc": first_time.isoformat() if first_time else None,
            "last_utc": last_time.isoformat() if last_time else None,
            "unique_observed_seconds": len(observed_seconds),
            "backward_timestamps": backward_timestamps,
            "duplicate_timestamps": duplicate_timestamps,
            "gaps_over_2s": gaps_over_2s,
            "largest_gap_seconds": largest_gap_seconds,
            "second_level_segments_gap_le_2s": segment_stats(ordered_observed_seconds),
            "local_days_with_observations": len(daily_counts),
        },
        "release_mapping": {
            "observed_seconds_during_any_release": len(during_values),
            "observed_seconds_outside_all_releases": len(background_values),
            "fraction_observed_seconds_during_release": len(during_values) / len(observed_seconds),
            "episodes_with_any_observation": sum(item["during_observed_seconds"] > 0 for item in episode_diagnostics),
            "episodes_with_pre10_and_first5_coverage": len(strict_episodes),
            "episodes_with_pre_during_post_context": len(complete_context_episodes),
            "episodes_with_at_least_60s_pre_and_30s_first5": sum(
                item["pre_10min_observed_seconds"] >= 60
                and item["first_5min_observed_seconds"] >= 30
                for item in episode_diagnostics
            ),
            "outside_release_segments_gap_le_2s": segment_stats(sorted(background_seconds)),
            "during_dry_ch4_ppm": quantiles(during_values),
            "outside_release_dry_ch4_ppm": background_quantiles,
        },
        "exploratory_signal_screen_not_final_evaluation": {
            "pooled_second_level_auc": pooled_auc,
            "daily_auc_days": len(finite_daily_auc),
            "daily_auc_quantiles": quantiles(finite_daily_auc),
            "thresholds_selected_on_all_outside_release_seconds": exploratory_thresholds,
            "warning": (
                "Second-level observations are temporally dependent and sparse; thresholds use the same "
                "data being summarized. These values only screen whether a signal exists and are not a "
                "train/test model result or event-level warning performance."
            ),
        },
        "daily_observed_seconds": {
            day: {
                "release": counts["release"],
                "outside_release": counts["outside_release"],
                "total": counts["release"] + counts["outside_release"],
            }
            for day, counts in sorted(daily_counts.items())
        },
        "episode_diagnostics": episode_diagnostics,
    }

    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
