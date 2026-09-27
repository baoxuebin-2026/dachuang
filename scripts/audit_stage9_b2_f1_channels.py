#!/usr/bin/env python3
"""Read-only, development-only physical channel audit; no fitting or evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "405a8b27004a2e318bf213cf209398d3d58b45bb3c7a78f159e69003ff459311"
START, END = "2014-03-02", "2014-04-30"
CHANNELS = ("MM256", "MM263", "MM264", "MM252", "MM261", "MM262", "MM211",
            "AN311", "AN422", "AN423", "TP1721", "RH1722", "BA1723",
            "TP1711", "RH1712", "BA1713", "WM868")
COLS = ("year", "month", "day", "hour", "minute", "second", *CHANNELS)
# Source article Table 1-3 measuring ranges. Other methane meter ranges are not specified there.
RANGES = {"AN311": (-5, 5), "AN422": (-5, 5), "AN423": (-5, 5),
          "RH1722": (0, 100), "RH1712": (0, 100), "WM868": (0, 50)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(a: np.ndarray) -> dict[str, float]:
    return dict(zip(("min", "p01", "p50", "p99", "max"),
                    map(float, np.quantile(a[np.isfinite(a)], [0, .01, .5, .99, 1]))))


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 100 or np.std(a[valid]) < 1e-10 or np.std(b[valid]) < 1e-10:
        return None
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def audit(archive_path: Path) -> dict:
    if sha256(archive_path) != SOURCE_SHA:
        raise ValueError("Source archive SHA256 mismatch")
    counts = {name: {"nonfinite": 0, "negative": 0, "zero": 0, "unchanged": 0,
                     "above_10_pct_if_methane": 0, "outside_documented_range": 0} for name in CHANNELS}
    prev = {name: None for name in CHANNELS}
    minutes: list[pd.DataFrame] = []
    rows = 0
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["methane_data.csv"] or archive.testzip() is not None:
            raise ValueError("Unexpected archive or CRC mismatch")
        with archive.open("methane_data.csv") as stream:
            for frame in pd.read_csv(stream, usecols=COLS, chunksize=150_000):
                days = pd.to_datetime(frame[["year", "month", "day"]]).dt.strftime("%Y-%m-%d")
                if days.iloc[0] > END:
                    break
                frame = frame.loc[days.between(START, END)]
                if frame.empty:
                    continue
                rows += len(frame)
                minutes.append(frame.loc[frame.second == 59, list(COLS)].copy())
                for name in CHANNELS:
                    a = frame[name].to_numpy(dtype=np.float64)
                    c = counts[name]
                    c["nonfinite"] += int((~np.isfinite(a)).sum())
                    c["negative"] += int((a < 0).sum())
                    c["zero"] += int((a == 0).sum())
                    if name.startswith("MM"):
                        c["above_10_pct_if_methane"] += int((a > 10).sum())
                    c["unchanged"] += int((a[1:] == a[:-1]).sum())
                    if prev[name] is not None:
                        c["unchanged"] += int(a[0] == prev[name])
                    prev[name] = a[-1]
                    if name in RANGES:
                        lo, hi = RANGES[name]
                        c["outside_documented_range"] += int(((a < lo) | (a > hi)).sum())
    if rows != 5_180_400:
        raise ValueError(f"Unexpected development rows: {rows}")
    minute = pd.concat(minutes, ignore_index=True)
    stamp = pd.to_datetime(minute[["year", "month", "day", "hour", "minute", "second"]])
    contiguous = (stamp.diff().dt.total_seconds().to_numpy() == 60)
    y = minute.MM256.to_numpy(dtype=float)
    result = {"status": "diagnostic_only; no model, labels, selection-period scoring, or causal claim",
              "source_archive_sha256": SOURCE_SHA, "development_dates": [START, END],
              "rows": rows, "minute_end_samples": len(minute), "channels": {},
              "relation_to_mm256": {}, "timestamp_noncontiguous_minute_edges": int((~contiguous[1:]).sum())}
    for name in CHANNELS:
        a = minute[name].to_numpy(dtype=float)
        c = counts[name]
        result["channels"][name] = {
            **c, "documented_range": RANGES.get(name), "quantiles_at_minute_end": quantiles(a),
            "zero_second_fraction": c["zero"] / rows,
            "unchanged_second_fraction": c["unchanged"] / (rows - 1),
            "distinct_minute_end_values": int(np.unique(a).size),
        }
        if name == "MM256":
            continue
        valid = np.isfinite(a) & np.isfinite(y)
        # Obvious invalid methane codes and physically invalid wind readings are excluded
        # from correlations, never silently repaired. No industrial threshold is invented.
        if name.startswith("MM"):
            valid &= (a >= 0) & (a <= 10)
        if name in ("AN311", "AN422", "AN423"):
            valid &= (a >= -5) & (a <= 5)
        valid &= (y >= 0) & (y <= 10)
        level = corr(a[valid], y[valid])
        rank = float(pd.Series(a[valid]).corr(pd.Series(y[valid]), method="spearman"))
        day = stamp.dt.strftime("%Y-%m-%d").to_numpy()
        daily = []
        for d in np.unique(day):
            mask = (day == d) & valid
            value = corr(a[mask], y[mask])
            if value is not None:
                daily.append(value)
        # First differences suppress shared day-long trends; channel differences at t
        # against target differences at t+lag are descriptive temporal association only.
        da, dy = np.diff(a), np.diff(y)
        good = valid[1:] & valid[:-1] & contiguous[1:]
        lead = {}
        for lag in (0, 1, 3, 5):
            x = da[:-lag] if lag else da
            z = dy[lag:]
            pair = (good[:-lag] & good[lag:]) if lag else good
            # Require every intervening one-minute edge to be contiguous.
            if lag:
                for offset in range(1, lag):
                    pair &= good[offset:offset-lag]
            lead[str(lag)] = corr(x[pair], z[pair])
        result["relation_to_mm256"][name] = {
            "valid_minute_pairs": int(valid.sum()), "pearson_levels": level,
            "spearman_levels": rank,
            "daily_pearson_median": float(np.median(daily)) if daily else None,
            "daily_pearson_p10_p90": list(map(float, np.quantile(daily, [.1, .9]))) if daily else None,
            "days_with_nonconstant_pairs": len(daily),
            "first_difference_channel_at_t_target_at_t_plus_minutes": lead,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage9_b2/f1_channel_audit_v1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": result["rows"], "minute_samples": result["minute_end_samples"]}))


if __name__ == "__main__":
    main()
