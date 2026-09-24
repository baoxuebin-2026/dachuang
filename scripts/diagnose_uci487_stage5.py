#!/usr/bin/env python3
"""Read-only UCI 487 diagnostics for day, 15-minute segment and heater phase.

python scripts/diagnose_uci487_stage5.py --archive ../uci_487_original.zip \
    --out results/uci487_protocol_diagnostic

This script fits no model and does not assign risk or accident labels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ["Time (s)", "CO (ppm)", "Humidity (%r.h.)", "Temperature (C)",
           "Flow rate (mL/min)", "Heater voltage (V)", "R1 (MOhm)", "R8 (MOhm)"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def segment_rows(frame: pd.DataFrame, day: str) -> list[dict]:
    time = frame["Time (s)"].to_numpy(dtype=float)
    block = np.floor((time - 900.0) / 900.0).astype(int)
    phase_s = (time - 900.0) - 900.0 * block
    mask = (block >= 0) & (block < 100) & (phase_s >= 600.0)
    settled = frame.loc[mask].copy()
    settled["segment"] = block[mask]
    settled["heater_high"] = settled["Heater voltage (V)"] > 0.55
    result = []
    for segment, part in settled.groupby("segment", sort=True):
        high = part.loc[part.heater_high]
        low = part.loc[~part.heater_high]
        if len(high) < 20 or len(low) < 20:
            raise ValueError(f"Insufficient heater phases: {day} segment {segment}")
        result.append({
            "day": day, "segment": int(segment), "source_kind": "UCI487_laboratory",
            "last_300_s_samples": len(part), "co_reference_ppm": float(part["CO (ppm)"].median()),
            "co_min_ppm": float(part["CO (ppm)"].min()),
            "co_max_ppm": float(part["CO (ppm)"].max()),
            "humidity_pct": float(part["Humidity (%r.h.)"].median()),
            "temperature_c": float(part["Temperature (C)"].median()),
            "heater_high_fraction": len(high) / len(part),
            "heater_high_n": len(high), "heater_low_n": len(low),
            "r1_high_mohm": float(high["R1 (MOhm)"].median()),
            "r1_low_mohm": float(low["R1 (MOhm)"].median()),
            "r8_high_mohm": float(high["R8 (MOhm)"].median()),
            "r8_low_mohm": float(low["R8 (MOhm)"].median()),
        })
    if len(result) != 100:
        raise ValueError(f"Expected 100 settled exposure segments, {day} has {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/uci487_protocol_diagnostic"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as outer:
        nested = outer.read("gas-sensor-array-temperature-modulation.zip")
    segments: list[dict] = []
    daily: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        names = sorted(name for name in inner.namelist() if name.endswith(".csv"))
        if len(names) != 13:
            raise ValueError(f"Expected 13 days, got {len(names)}")
        for name in names:
            day = Path(name).stem
            frame = pd.read_csv(inner.open(name), usecols=COLUMNS)
            if frame.shape[1] != len(COLUMNS) or not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError(f"Unexpected columns/nonfinite measurement: {day}")
            current = segment_rows(frame, day)
            segments.extend(current)
            daysegments = pd.DataFrame(current)
            co = daysegments.co_reference_ppm.round(2)
            levels = co.value_counts()
            zero = daysegments.loc[co == 0.0]
            daily.append({
                "day": day, "rows": len(frame), "segments": len(current),
                "co_levels": int(len(levels)), "level_repetitions_min": int(levels.min()),
                "level_repetitions_max": int(levels.max()),
                "co_range_ppm": f"{co.min():g}..{co.max():g}",
                "humidity_segment_min_pct": float(daysegments.humidity_pct.min()),
                "humidity_segment_max_pct": float(daysegments.humidity_pct.max()),
                "co_humidity_segment_pearson": float(daysegments.co_reference_ppm.corr(daysegments.humidity_pct)),
                "zero_co_segments": int(len(zero)),
                "zero_co_r1_low_median_mohm": float(zero.r1_low_mohm.median()),
                "r1_phase_difference_abs_median_mohm": float((daysegments.r1_high_mohm - daysegments.r1_low_mohm).abs().median()),
                "r8_phase_difference_abs_median_mohm": float((daysegments.r8_high_mohm - daysegments.r8_low_mohm).abs().median()),
                "heater_high_fraction_mean": float(daysegments.heater_high_fraction.mean()),
                "backward_time_steps": int((np.diff(frame["Time (s)"].to_numpy()) < 0).sum()),
                "heater_outside_0_1_v": int(((frame["Heater voltage (V)"] < 0)
                                              | (frame["Heater voltage (V)"] > 1)).sum()),
            })
    segment_table = pd.DataFrame(segments)
    day_table = pd.DataFrame(daily)
    co_grid = segment_table.pivot(index="segment", columns="day", values="co_reference_ppm")
    rh_grid = segment_table.pivot(index="segment", columns="day", values="humidity_pct")
    first_day = co_grid.iloc[:, 0]
    same_co = (co_grid.round(2).eq(first_day.round(2), axis=0)).sum(axis=0)
    max_co_difference = float(np.max(np.abs(co_grid.to_numpy() - first_day.to_numpy()[:, None])))
    rh_pairs = rh_grid.corr().to_numpy()
    rh_pairs = rh_pairs[np.triu_indices(len(rh_grid.columns), 1)]
    segment_table.to_csv(args.out / "segment_diagnostic.csv", index=False, lineterminator="\n")
    day_table.to_csv(args.out / "day_diagnostic.csv", index=False, lineterminator="\n")
    summary = {
        "validation_status": "diagnostic_only_no_model", "dataset": "UCI_487_laboratory",
        "archive_sha256": sha256(args.archive), "script_sha256": sha256(Path(__file__)),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "days": len(day_table), "segments": len(segment_table),
        "co_levels_each_day": day_table.co_levels.tolist(),
        "repeat_min_max_across_days": [int(day_table.level_repetitions_min.min()),
                                        int(day_table.level_repetitions_max.max())],
        "co_humidity_correlation_range": [float(day_table.co_humidity_segment_pearson.min()),
                                           float(day_table.co_humidity_segment_pearson.max())],
        "r1_high_low_phase_abs_median_range_mohm": [
            float(day_table.r1_phase_difference_abs_median_mohm.min()),
            float(day_table.r1_phase_difference_abs_median_mohm.max())],
        "zero_co_r1_low_median_range_mohm": [float(day_table.zero_co_r1_low_median_mohm.min()),
                                              float(day_table.zero_co_r1_low_median_mohm.max())],
        "segment_humidity_pct_range": [float(segment_table.humidity_pct.min()),
                                        float(segment_table.humidity_pct.max())],
        "heater_high_fraction_range": [float(segment_table.heater_high_fraction.min()),
                                        float(segment_table.heater_high_fraction.max())],
        "time_backsteps": int(day_table.backward_time_steps.sum()),
        "same_co_position_count_vs_first_day_min_max": [int(same_co.min()), int(same_co.max())],
        "co_reference_max_position_difference_vs_first_day_ppm": max_co_difference,
        "humidity_sequence_pairwise_pearson_range": [float(rh_pairs.min()), float(rh_pairs.max())],
        "clock_lookup_note": "The 100-slot first-day CO reference is an invalid but exact CO predictor for every other day at settled segments",
        "segment_sample_count_range": [int(segment_table.last_300_s_samples.min()),
                                        int(segment_table.last_300_s_samples.max())],
        "target_note": "Dataset CO reference in ppm from chamber gas generation/measurement; not independent field accident labels",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
