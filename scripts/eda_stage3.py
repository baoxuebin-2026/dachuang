#!/usr/bin/env python3
"""Read-only EDA for UCI 309 and 487; outputs are diagnostic, not model results.

Run from the repository root with the official, unchanged UCI ZIP files:
python scripts/eda_stage3.py --uci309 /path/uci_309_original.zip \
    --uci487 /path/uci_487_original.zip --out outputs/eda_stage3

Needs numpy, pandas and matplotlib. No raw rows are written to the output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAME_309 = re.compile(r"^(\d{3})_Et_([nLMH])_(CO|Me)_([nLMH])$")
COLS_309 = ["time_s", "temperature_c", "humidity_pct"] + [
    f"sensor_{i}_adc" for i in range(1, 9)
]
PALETTE = ("#4E79A7", "#F28E2B", "#59A14F", "#D95F5F")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def finite_summary(frame: pd.DataFrame, time_col: str) -> dict:
    values = frame[time_col].to_numpy(dtype=float)
    diffs = np.diff(values)
    return {
        "rows": len(frame),
        "missing_cells": int(frame.isna().to_numpy().sum()),
        "nonfinite_cells": int(np.count_nonzero(~np.isfinite(frame.to_numpy(dtype=float)))),
        "first_s": float(values[0]),
        "last_s": float(values[-1]),
        "duplicate_timestamps": int(np.count_nonzero(diffs == 0)),
        "backward_timestamps": int(np.count_nonzero(diffs < 0)),
        "median_positive_step_s": float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else None,
        "maximum_step_s": float(diffs.max()) if len(diffs) else None,
    }


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def investigate_309(archive: Path, out: Path) -> dict:
    records: list[dict] = []
    curves: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(archive) as z:
        names = [
            x for x in z.namelist() if x.startswith("dataset_twosources_") and not x.endswith("/")
        ]
        for name in sorted(names):
            kind, base = name.split("/", 1)
            match = NAME_309.fullmatch(base)
            if not match:
                raise ValueError(f"Unexpected UCI 309 file name: {name}")
            file_id, eth_level, gas, second_level = match.groups()
            frame = pd.read_csv(z.open(name), header=None, names=COLS_309)
            if frame.shape[1] != 11:
                raise ValueError(f"Incorrect UCI 309 columns: {name}")
            stats = finite_summary(frame, "time_s")
            values = frame[[f"sensor_{i}_adc" for i in range(1, 9)]].to_numpy()
            stats.update(
                file_id=file_id,
                resolution=kind.replace("dataset_twosources_", ""),
                ethylene_level=eth_level,
                second_gas=gas,
                second_level=second_level,
                below_or_equal_zero_adc=int((values <= 0).sum()),
                above_or_equal_3110_adc=int((values >= 3110).sum()),
                temperature_min_c=float(frame.temperature_c.min()),
                temperature_max_c=float(frame.temperature_c.max()),
                humidity_min_pct=float(frame.humidity_pct.min()),
                humidity_max_pct=float(frame.humidity_pct.max()),
                background_rows=int((frame.time_s < 60).sum()),
                exposure_rows=int(((frame.time_s >= 60) & (frame.time_s < 240)).sum()),
                recovery_rows=int((frame.time_s >= 240).sum()),
            )
            if stats["resolution"] == "downsampled":
                background = frame.loc[(frame.time_s >= 20) & (frame.time_s < 55), "sensor_1_adc"]
                exposure = frame.loc[(frame.time_s >= 120) & (frame.time_s < 230), "sensor_1_adc"]
                stats["sensor1_relative_exposure_change"] = float(exposure.median() / background.median() - 1)
                early = frame.loc[(frame.time_s >= 10) & (frame.time_s < 30), "sensor_1_adc"]
                late = frame.loc[(frame.time_s >= 35) & (frame.time_s < 55), "sensor_1_adc"]
                stats["sensor1_prerelease_relative_drift"] = float(late.median() / early.median() - 1)
            records.append(stats)
            if stats["resolution"] == "downsampled" and file_id in {"001", "002"}:
                curves[file_id] = frame
    table = pd.DataFrame(records)
    table.to_csv(out / "uci309_file_inventory.csv", index=False)
    down = table[table.resolution == "downsampled"]
    raw = table[table.resolution == "raw"]
    counts = down.groupby(["ethylene_level", "second_gas", "second_level"]).size()
    fig, axs = plt.subplots(2, 1, figsize=(7.1, 5.1), sharex=True)
    for ax, key, label in zip(axs, ("001", "002"), ("001: ethylene L + methane H", "002: ethylene H + CO H")):
        frame = curves[key]
        baseline = frame.loc[(frame.time_s >= 20) & (frame.time_s < 55), "sensor_1_adc"].median()
        ax.plot(frame.time_s, frame.sensor_1_adc / baseline - 1, lw=0.8, color=PALETTE[0])
        ax.axvline(60, lw=0.8, ls="--", color=PALETTE[1])
        ax.axvline(240, lw=0.8, ls="--", color=PALETTE[2])
        ax.set_ylabel("Sensor 1 / baseline - 1")
        ax.text(0.02, 0.91, label, transform=ax.transAxes, fontsize=9)
        ax.grid(axis="y", alpha=0.22)
    axs[-1].set_xlabel("Relative time (s)")
    fig.tight_layout()
    save_figure(fig, out / "uci309_representative_diagnostic")
    return {
        "dataset": "UCI 309",
        "kind": "laboratory_measurements",
        "raw_files": len(raw),
        "downsampled_files": len(down),
        "raw_rows": int(raw.rows.sum()),
        "downsampled_rows": int(down.rows.sum()),
        "gas_groups": down.groupby("second_gas").size().astype(int).to_dict(),
        "configuration_counts": {
            "number": int(counts.size),
            "minimum_repetitions": int(counts.min()),
            "maximum_repetitions": int(counts.max()),
        },
        "both_sources_zero_files": int(((down.ethylene_level == "n") & (down.second_level == "n")).sum()),
        "missing_cells": int(table.missing_cells.sum()),
        "nonfinite_cells": int(table.nonfinite_cells.sum()),
        "duplicate_timestamps": {
            "raw_total": int(raw.duplicate_timestamps.sum()),
            "downsampled_total": int(down.duplicate_timestamps.sum()),
            "downsampled_affected_files": int((down.duplicate_timestamps > 0).sum()),
        },
        "maximum_gap_s": {
            "raw": float(raw.maximum_step_s.max()),
            "downsampled": float(down.maximum_step_s.max()),
        },
        "time_start_range_s": [float(down.first_s.min()), float(down.first_s.max())],
        "time_end_range_s": [float(down.last_s.min()), float(down.last_s.max())],
        "sensor_adc_invalid_count": int(table.below_or_equal_zero_adc.sum() + table.above_or_equal_3110_adc.sum()),
        "temperature_range_c": [float(table.temperature_min_c.min()), float(table.temperature_max_c.max())],
        "humidity_range_pct": [float(table.humidity_min_pct.min()), float(table.humidity_max_pct.max())],
        "downsampled_background_rows_range": [int(down.background_rows.min()), int(down.background_rows.max())],
        "sensor1_relative_exposure_change_quantiles": {
            str(q): float(down.sensor1_relative_exposure_change.quantile(q))
            for q in (0.05, 0.5, 0.95)
        },
        "sensor1_prerelease_drift_quantiles": {
            str(q): float(down.sensor1_prerelease_relative_drift.quantile(q))
            for q in (0.05, 0.5, 0.95)
        },
    }


def investigate_487(archive: Path, out: Path) -> dict:
    daily: list[dict] = []
    first_curve: pd.DataFrame | None = None
    with zipfile.ZipFile(archive) as outer:
        inner_bytes = outer.read("gas-sensor-array-temperature-modulation.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        names = sorted(x for x in inner.namelist() if x.endswith(".csv"))
        for name in names:
            frame = pd.read_csv(inner.open(name), skipinitialspace=True)
            if frame.shape[1] != 20:
                raise ValueError(f"Incorrect UCI 487 columns: {name}")
            cols = frame.columns
            stats = finite_summary(frame, cols[0])
            gas, rh, temp, flow, heater = [frame[x] for x in cols[1:6]]
            resistance = frame.iloc[:, 6:20]
            stimulus = frame.loc[(frame[cols[0]] >= 900) & (frame[cols[0]] < 90900)]
            window = ((stimulus.iloc[:, 0] - 900) // 900).astype(int)
            plateau_co = stimulus.groupby(window)[cols[1]].median().round(2)
            plateau_counts = plateau_co.value_counts()
            stats.update(
                day_file=name,
                co_min_ppm=float(gas.min()),
                co_max_ppm=float(gas.max()),
                co_positive_fraction=float((gas > 0).mean()),
                co_unique_values=int(gas.nunique()),
                humidity_min_pct=float(rh.min()),
                humidity_max_pct=float(rh.max()),
                temperature_min_c=float(temp.min()),
                temperature_max_c=float(temp.max()),
                heater_min_v=float(heater.min()),
                heater_max_v=float(heater.max()),
                flow_min_ml_min=float(flow.min()),
                flow_max_ml_min=float(flow.max()),
                sensor_resistance_nonpositive_cells=int((resistance <= 0).to_numpy().sum()),
                co_negative_rows=int((gas < 0).sum()),
                humidity_outside_0_100_rows=int(((rh < 0) | (rh > 100)).sum()),
                humidity_above_protocol_75_rows=int((rh > 75).sum()),
                temperature_below_15_rows=int((temp < 15).sum()),
                flow_zero_rows=int((flow == 0).sum()),
                sensor1_co_pearson=float(stimulus.iloc[:, 6].corr(stimulus.iloc[:, 1])),
                sensor1_humidity_pearson=float(stimulus.iloc[:, 6].corr(stimulus.iloc[:, 2])),
                sensor1_heater_pearson=float(stimulus.iloc[:, 6].corr(stimulus.iloc[:, 5])),
                plateau_window_count=len(plateau_co),
                plateau_co_level_count=len(plateau_counts),
                plateau_level_repetitions_min=int(plateau_counts.min()),
                plateau_level_repetitions_max=int(plateau_counts.max()),
            )
            daily.append(stats)
            if name == names[0]:
                first_curve = frame.loc[frame.iloc[:, 0] < 3600].iloc[::17].copy()
    table = pd.DataFrame(daily)
    table.to_csv(out / "uci487_day_inventory.csv", index=False)
    if first_curve is None:
        raise ValueError("UCI 487 archive contains no day files")
    fig, axs = plt.subplots(3, 1, figsize=(7.1, 6.2), sharex=True)
    hrs = first_curve.iloc[:, 0].to_numpy() / 3600
    for ax, col, color, unit in zip(
        axs,
        [first_curve.columns[1], first_curve.columns[2], first_curve.columns[6]],
        [PALETTE[0], PALETTE[1], PALETTE[2]],
        ["CO (ppm)", "Relative humidity (%)", "Sensor R1 (MOhm)"],
    ):
        ax.plot(hrs, first_curve[col], color=color, lw=0.9)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.22)
    axs[-1].set_xlabel("Elapsed time on first measurement day (h)")
    fig.tight_layout()
    save_figure(fig, out / "uci487_first_hour_diagnostic")
    return {
        "dataset": "UCI 487",
        "kind": "laboratory_measurements",
        "measurement_days": len(table),
        "rows": int(table.rows.sum()),
        "day_rows_range": [int(table.rows.min()), int(table.rows.max())],
        "day_duration_hours_range": [float(table.last_s.min() / 3600), float(table.last_s.max() / 3600)],
        "missing_cells": int(table.missing_cells.sum()),
        "nonfinite_cells": int(table.nonfinite_cells.sum()),
        "duplicate_timestamps": int(table.duplicate_timestamps.sum()),
        "backward_timestamps": int(table.backward_timestamps.sum()),
        "maximum_step_s": float(table.maximum_step_s.max()),
        "co_range_ppm": [float(table.co_min_ppm.min()), float(table.co_max_ppm.max())],
        "humidity_range_pct": [float(table.humidity_min_pct.min()), float(table.humidity_max_pct.max())],
        "temperature_range_c": [float(table.temperature_min_c.min()), float(table.temperature_max_c.max())],
        "heater_range_v": [float(table.heater_min_v.min()), float(table.heater_max_v.max())],
        "flow_range_ml_min": [float(table.flow_min_ml_min.min()), float(table.flow_max_ml_min.max())],
        "sensor_resistance_nonpositive_cells": int(table.sensor_resistance_nonpositive_cells.sum()),
        "negative_co_rows": int(table.co_negative_rows.sum()),
        "humidity_outside_0_100_rows": int(table.humidity_outside_0_100_rows.sum()),
        "humidity_above_protocol_75_rows": int(table.humidity_above_protocol_75_rows.sum()),
        "temperature_below_15_rows": int(table.temperature_below_15_rows.sum()),
        "flow_zero_rows": int(table.flow_zero_rows.sum()),
        "sensor1_absolute_heater_correlation_range_by_day": [
            float(table.sensor1_heater_pearson.abs().min()),
            float(table.sensor1_heater_pearson.abs().max()),
        ],
        "plateau_window_count_range_by_day": [
            int(table.plateau_window_count.min()),
            int(table.plateau_window_count.max()),
        ],
        "plateau_co_level_count_range_by_day": [
            int(table.plateau_co_level_count.min()),
            int(table.plateau_co_level_count.max()),
        ],
        "plateau_level_repetitions_range_by_day": [
            int(table.plateau_level_repetitions_min.min()),
            int(table.plateau_level_repetitions_max.max()),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uci309", required=True, type=Path)
    parser.add_argument("--uci487", required=True, type=Path)
    parser.add_argument("--out", default=Path("outputs/eda_stage3"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "validation_status": "diagnostic_only",
        "script_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "data_archives": {
            "uci309_sha256": sha256(args.uci309),
            "uci487_sha256": sha256(args.uci487),
        },
        "uci309": investigate_309(args.uci309, args.out),
        "uci487": investigate_487(args.uci487, args.out),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
