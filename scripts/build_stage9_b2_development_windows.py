#!/usr/bin/env python3
"""Construct B2-B proxy events and minute windows on development dates only.

No model is fitted. Only the target channel is read; labels and event metadata
are segregated from any future model features. Both processed CSVs are ignored
by Git. Published seconds were already aligned/forward-filled upstream.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage9_b2_protocol_v1.json"
ARCHIVE = ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip"
OUTPUT = ROOT / "results/stage9_b2/development_windows_v1.json"
PROCESSED = ROOT / "data/processed/stage9_b2"
TIME = ["year", "month", "day", "hour", "minute", "second"]
TARGETS = ["MM256", "MM263", "MM264"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development(path: Path, config: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if file_sha256(path) != config["source_sha256"]:
        raise ValueError("Source archive SHA-256 mismatch")
    lo, hi = config["development_dates_inclusive"]
    stamps: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    channels: dict[str, list[np.ndarray]] = {name: [] for name in TARGETS}
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["methane_data.csv"] or archive.testzip() is not None:
            raise ValueError("Archive member or CRC mismatch")
        with archive.open("methane_data.csv") as stream:
            for frame in pd.read_csv(stream, usecols=[*TIME, *TARGETS], chunksize=150_000):
                dates = pd.to_datetime(frame[["year", "month", "day"]]).dt.strftime("%Y-%m-%d")
                if dates.iloc[0] > hi:
                    break  # Never parse later raw values in this script.
                keep = dates.between(lo, hi)
                if not keep.any():
                    continue
                frame = frame.loc[keep]
                stamps.append(pd.to_datetime(frame[TIME]).to_numpy(dtype="datetime64[s]").astype(np.int64))
                seconds.append(frame["second"].to_numpy(dtype=np.int8))
                for name in TARGETS:
                    channels[name].append(frame[name].to_numpy(dtype=np.float32))
    t = np.concatenate(stamps)
    s = np.concatenate(seconds)
    x = {name: np.concatenate(parts) for name, parts in channels.items()}
    if len(t) != 5_180_400 or any(len(a) != len(t) for a in x.values()):
        raise ValueError(f"Unexpected development row count: {len(t)}")
    return t, s, x


def prefix(mask: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(mask.astype(np.int32), dtype=np.int32)]


def audit_channel(timestamps: np.ndarray, seconds: np.ndarray, values: np.ndarray,
                  config: dict) -> tuple[dict, list[dict], list[dict]]:
    """Return count audit, event rows, eligible window rows; no fitted quantities."""
    n = len(timestamps)
    threshold = config["threshold_pct_ch4"]
    quiet = config["event_prior_quiet_seconds"]
    sustained = config["event_min_consecutive_seconds"]
    history = config["prediction_history_seconds"]
    horizon = config["prediction_horizon_seconds"]
    extra = config["label_confirmation_extra_seconds"]
    valid = np.isfinite(values) & (values >= config["valid_reading_min_inclusive"]) & (
        values <= config["valid_reading_max_inclusive"])
    low = valid & (values < threshold)
    high = valid & (values >= threshold)
    gap = np.r_[False, np.diff(timestamps) != 1]
    bad_cum = prefix(~valid)
    notlow_cum = prefix(~low)
    gap_cum = np.cumsum(gap.astype(np.int32), dtype=np.int32)

    continues = high[1:] & high[:-1] & (~gap[1:])
    starts = np.flatnonzero(high & ~np.r_[False, continues])
    ends = np.flatnonzero(high & ~np.r_[continues, False]) + 1  # exclusive
    lengths = ends - starts
    sustained_mask = lengths >= sustained
    preceding = starts >= quiet
    preceding[preceding] &= (
        (notlow_cum[starts[preceding]] - notlow_cum[starts[preceding] - quiet] == 0)
        & (gap_cum[starts[preceding]] - gap_cum[starts[preceding] - quiet] == 0)
    )
    event_start = starts[sustained_mask & preceding]
    event_end = ends[sustained_mask & preceding]
    anchor = np.flatnonzero(seconds == config["prediction_second_of_minute"])
    reasons = np.full(len(anchor), "eligible", dtype="U24")

    a = anchor - history
    has_past = a >= 0
    reasons[~has_past] = "insufficient_history"
    ix = np.flatnonzero(has_past)
    past_bad = ((bad_cum[anchor[ix] + 1] - bad_cum[a[ix]] != 0)
                | (gap_cum[anchor[ix]] - gap_cum[a[ix]] != 0))
    reasons[ix[past_bad]] = "invalid_history"

    ix = np.flatnonzero(reasons == "eligible")
    not_quiet = notlow_cum[anchor[ix] + 1] - notlow_cum[anchor[ix] - quiet] != 0
    reasons[ix[not_quiet]] = "not_quiet_180"

    ix = np.flatnonzero(reasons == "eligible")
    b = anchor[ix] + horizon + extra
    reasons[ix[b >= n]] = "incomplete_future"
    ix = np.flatnonzero(reasons == "eligible")
    b = anchor[ix] + horizon + extra
    future_bad = ((bad_cum[b + 1] - bad_cum[anchor[ix] + 1] != 0)
                  | (gap_cum[b] - gap_cum[anchor[ix]] != 0))
    reasons[ix[future_bad]] = "invalid_future"

    eligible = anchor[reasons == "eligible"]
    next_idx = np.searchsorted(event_start, eligible, side="right")
    positive = next_idx < len(event_start)
    positive[positive] = event_start[next_idx[positive]] <= eligible[positive] + horizon

    event_rows: list[dict] = []
    for onset, end in zip(event_start, event_end):
        left = np.searchsorted(eligible, onset - horizon, side="left")
        right = np.searchsorted(eligible, onset, side="left")
        event_rows.append({
            "source_row": int(onset), "onset_local_naive": str(np.datetime64(int(timestamps[onset]), "s")),
            "duration_seconds": int(end - onset), "eligible_pre_event_anchors": int(right - left),
            "earliest_eligible_lead_seconds": int(timestamps[onset] - timestamps[eligible[left]]) if right > left else None,
            "latest_eligible_lead_seconds": int(timestamps[onset] - timestamps[eligible[right - 1]]) if right > left else None,
        })
    window_rows = [{
        "source_row": int(row), "prediction_local_naive": str(np.datetime64(int(timestamps[row]), "s")),
        "proxy_label": int(label),
    } for row, label in zip(eligible, positive)]
    counts = Counter(reasons.tolist())
    summary = {
        "target": "assigned_by_caller", "development_seconds": n,
        "raw_crossing_runs": int(len(starts)),
        "runs_at_least_30_seconds": int(sustained_mask.sum()),
        "clean_sustained_proxy_events": int(len(event_rows)),
        "evaluable_proxy_events": sum(bool(row["eligible_pre_event_anchors"]) for row in event_rows),
        "total_minute_anchors": int(len(anchor)),
        "anchor_status": dict(sorted(counts.items())),
        "positive_windows": int(positive.sum()),
        "negative_windows": int(len(positive) - positive.sum()),
        "timestamp_discontinuities": int(gap.sum()),
        "invalid_target_seconds": int((~valid).sum()),
        "interpretation": "Only development proxy-label coverage; not accuracy, independent accidents, or fitted-model performance.",
    }
    assert sum(counts.values()) == len(anchor)
    assert summary["positive_windows"] + summary["negative_windows"] == counts["eligible"]
    return summary, event_rows, window_rows


def write_gzip_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(args.processed_dir.joinpath(f"development_{name}_{kind}.csv.gz").exists()
           for name in TARGETS for kind in ("events", "windows")):
        raise FileExistsError("Existing processed B2 development files")
    t, s, channels = load_development(args.archive, config)
    report = {
        "status": "development_proxy_labels_only; no model or reserved evaluation",
        "protocol_id": config["protocol_id"], "protocol_sha256": file_sha256(CONFIG),
        "source_archive_sha256": config["source_sha256"],
        "targets": {}, "processed_files_ignored_by_git": [],
    }
    computed = {}
    for name in TARGETS:
        summary, events, windows = audit_channel(t, s, channels[name], config)
        summary["target"] = name
        report["targets"][name] = summary
        computed[name] = (events, windows)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, (events, windows) in computed.items():
        for kind, rows, fields in (
            ("events", events, ["source_row", "onset_local_naive", "duration_seconds",
                               "eligible_pre_event_anchors", "earliest_eligible_lead_seconds",
                               "latest_eligible_lead_seconds"]),
            ("windows", windows, ["source_row", "prediction_local_naive", "proxy_label"]),
        ):
            path = args.processed_dir / f"development_{name}_{kind}.csv.gz"
            write_gzip_csv(path, rows, fields)
            report["processed_files_ignored_by_git"].append({"path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path), "rows": len(rows)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["targets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
