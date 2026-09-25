#!/usr/bin/env python3
"""UCI 487: blocked-day/blocked-position CO calibration at completed 15-min segments.

python scripts/run_uci487_v3.py --archive data/raw/uci_487_original.zip \
    --out results/uci487_v3

The reference CO, clock and segment number are for splitting/scoring only.
All model features are segment-local sensor/heater/ambient measurements.
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
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SENSORS = [f"R{i} (MOhm)" for i in range(1, 15)]
INPUT_COLS = ["Time (s)", "CO (ppm)", "Humidity (%r.h.)", "Temperature (C)",
              "Heater voltage (V)", *SENSORS]
SEED = 20260924
ALPHAS = (0.1, 1.0, 10.0, 100.0)
TARGET = "co_reference_ppm"
AMBIENT = ("humidity_pct", "temperature_c")
BLIND = tuple(f"log_r{i}_blind" for i in range(1, 15))
PHASE = tuple(f"log_r{i}_{phase}" for i in range(1, 15) for phase in ("low", "high"))
MODELS = {
    "ambient_only": AMBIENT,
    "r1_blind": (BLIND[0],),
    "all14_blind": BLIND,
    "all14_phase": PHASE,
    "all14_phase_ambient": (*PHASE, *AMBIENT),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_manifest(days: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    """Use labels only for offline, within-level assignment of whole positions."""
    if len(days) != 13 or reference.shape[0] != 100:
        raise ValueError("Expected 13 days with 100 reference positions")
    levels = reference.set_index("segment")[TARGET]
    if levels.nunique() != 10 or not (levels.value_counts() == 10).all():
        raise ValueError("Expected 10 CO levels, 10 positions per level")
    rng = np.random.default_rng(SEED)
    assignments = {}
    for _, positions in levels.groupby(levels, sort=True):
        shuffled = rng.permutation(positions.index.to_numpy(dtype=int))
        assignments.update({int(p): "train" for p in shuffled[:6]})
        assignments.update({int(p): "validation" for p in shuffled[6:8]})
        assignments.update({int(p): "test" for p in shuffled[8:]})
    records = []
    for i, day in enumerate(days):
        allowed = "train" if i < 8 else "validation" if i < 10 else "test"
        for position in range(100):
            position_split = assignments[position]
            records.append({"day": day, "segment": position, "co_level_offline_ppm": levels[position],
                            "position_split": position_split, "day_split": allowed,
                            "split": allowed if allowed == position_split else "excluded_cross_cell"})
    manifest = pd.DataFrame(records)
    assert manifest.split.value_counts().to_dict() == {
        "excluded_cross_cell": 720, "train": 480, "validation": 40, "test": 60,
    }
    sets = {split: set(manifest.loc[manifest.split == split, "segment"])
            for split in ("train", "validation", "test")}
    assert all(not sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"),
                                                    ("validation", "test")))
    assert {key: len(positions) for key, positions in sets.items()} == {
        "train": 60, "validation": 20, "test": 20,
    }
    return manifest


def extract_day(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != tuple(INPUT_COLS) or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"Unexpected columns or nonfinite measurement in {name}")
    time = frame["Time (s)"].to_numpy()
    position = np.floor((time - 900.0) / 900.0).astype(int)
    since_start = (time - 900.0) - position * 900.0
    good = (position >= 0) & (position < 100) & (since_start >= 600.0)
    frame = frame.loc[good].copy()
    frame["segment"] = position[good]
    # 0.55 V is the midpoint of the two documented 0.2/0.9 V heater states.
    frame["phase"] = np.where(frame["Heater voltage (V)"] > 0.55, "high", "low")
    if (frame[SENSORS].to_numpy() <= 0).any():
        raise ValueError(f"Cannot log-transform nonpositive resistance in {name}")
    names = {r: f"log_r{i}_blind" for i, r in enumerate(SENSORS, 1)}
    for r, dest in names.items():
        frame[dest] = np.log(frame[r])
    b = frame.groupby("segment", sort=True)
    blind = b[list(BLIND)].median()
    label = b["CO (ppm)"].agg(["median", "min", "max"]).rename(
        columns={"median": TARGET, "min": "co_reference_min_ppm", "max": "co_reference_max_ppm"})
    ambient = b[["Humidity (%r.h.)", "Temperature (C)"]].median().rename(
        columns={"Humidity (%r.h.)": AMBIENT[0], "Temperature (C)": AMBIENT[1]})
    phase = frame.groupby(["segment", "phase"], sort=True)[list(BLIND)].median().unstack("phase")
    if any((r, p) not in phase.columns for r in BLIND for p in ("low", "high")):
        raise ValueError(f"Missing heater phase in {name}")
    phase.columns = [f"{r[:-6]}_{p}" for r, p in phase.columns]
    counts = frame.groupby(["segment", "phase"], sort=True).size().unstack(fill_value=0)
    if (counts[["high", "low"]] < 20).any().any():
        raise ValueError(f"Under 20 samples in a heater phase in {name}")
    result = pd.concat([label, ambient, blind, phase], axis=1).reset_index()
    result.insert(0, "day", name)
    if result.shape[0] != 100 or result.isna().any().any():
        raise ValueError(f"Missing/duplicated 15-minute segment in {name}")
    return result


def load_features(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as source:
        with zipfile.ZipFile(io.BytesIO(source.read("gas-sensor-array-temperature-modulation.zip"))) as inner:
            files = sorted(n for n in inner.namelist() if n.endswith(".csv"))
            if len(files) != 13:
                raise ValueError("Expected 13 measurement days")
            frames = []
            for path in files:
                frame = pd.read_csv(inner.open(path), usecols=INPUT_COLS)
                frames.append(extract_day(Path(path).stem, frame))
                print(f"Parsed {Path(path).stem}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite derived features")
    return result


def fit_candidates(train: pd.DataFrame, val: pd.DataFrame):
    """Fit/tune on train/validation, seal choices before the test prediction."""
    baseline = float(train[TARGET].median())
    selections = [{"model": "constant_train_median", "alpha": "NA",
                   "validation_mae_ppm": float(mean_absolute_error(val[TARGET],
                                                                    np.full(len(val), baseline)))}]
    fitted = {"constant_train_median": (None, (), baseline)}
    for model, cols in MODELS.items():
        if set(cols) & {TARGET, "segment", "day", "co_level_offline_ppm", "co_reference_min_ppm",
                        "co_reference_max_ppm", "split", "position_split", "day_split"}:
            raise ValueError(f"Forbidden feature in {model}")
        choices = []
        for alpha in ALPHAS:
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipe.fit(train[list(cols)], train[TARGET])
            error = float(mean_absolute_error(val[TARGET], pipe.predict(val[list(cols)])))
            choices.append((error, alpha, pipe))
        best_mae, best_alpha, best_pipe = min(choices, key=lambda row: (row[0], row[1]))
        fitted[model] = (best_pipe, cols, None)
        selections.append({"model": model, "alpha": best_alpha, "validation_mae_ppm": best_mae})
    return fitted, pd.DataFrame(selections)


def predictions(frame: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    rows = []
    for model, (estimator, cols, baseline) in fitted.items():
        predicted = (np.full(len(frame), baseline) if estimator is None
                     else estimator.predict(frame[list(cols)]))
        for day, position, truth, pred in zip(frame.day, frame.segment, frame[TARGET], predicted):
            rows.append({"split": frame.split.iloc[0], "model": model, "day": day,
                         "segment": int(position), "co_reference_ppm": float(truth),
                         "predicted_ppm": float(pred), "absolute_error_ppm": float(abs(truth - pred))})
    return pd.DataFrame(rows)


def daily_v1(features: pd.DataFrame) -> pd.DataFrame:
    records = []
    for day, block in features.groupby("day", sort=True):
        zero = block.loc[block[TARGET] == 0]
        if len(zero) != 10:
            raise ValueError(f"Unexpected zero-CO segments: {day}")
        records.append({"day": day, "zero_co_segments": len(zero),
                        "zero_co_r1_low_mohm": float(np.median(np.exp(zero.log_r1_low))),
                        "zero_co_r8_low_mohm": float(np.median(np.exp(zero.log_r8_low))),
                        "r1_phase_difference_abs_mohm": float(np.median(
                            abs(np.exp(block.log_r1_low) - np.exp(block.log_r1_high)))),
                        "r8_phase_difference_abs_mohm": float(np.median(
                            abs(np.exp(block.log_r8_low) - np.exp(block.log_r8_high)))),
                        "humidity_median_pct": float(block.humidity_pct.median()),
                        "humidity_min_pct": float(block.humidity_pct.min()),
                        "humidity_max_pct": float(block.humidity_pct.max())})
    return pd.DataFrame(records)


def aggregate_error(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = []
    for group, part in rows.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        result.append({**dict(zip(keys, group)), "n": len(part),
                       "mae_ppm": float(mean_absolute_error(part.co_reference_ppm, part.predicted_ppm)),
                       "rmse_ppm": float(np.sqrt(mean_squared_error(part.co_reference_ppm,
                                                                     part.predicted_ppm))),
                       "bias_ppm": float(np.mean(part.predicted_ppm - part.co_reference_ppm))})
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/uci487_v3"))
    args = parser.parse_args()
    features = load_features(args.archive)
    days = sorted(features.day.unique())
    table = features.pivot(index="segment", columns="day", values=TARGET)
    if not table.eq(table.iloc[:, 0], axis=0).all().all():
        raise ValueError("CO program differs across days: revisit the split protocol")
    manifest = make_manifest(days, features.loc[features.day == days[0]])
    features = features.merge(manifest, on=["day", "segment"], validate="one_to_one")
    if not (features[TARGET] == features.co_level_offline_ppm).all():
        raise ValueError("Day and offline CO level differ")
    train = features.loc[features.split == "train"].copy()
    val = features.loc[features.split == "validation"].copy()
    test = features.loc[features.split == "test"].copy()
    assert (len(train), len(val), len(test)) == (480, 40, 60)
    assert len(set(train.day) & set(test.day)) == 0
    fitted, selection = fit_candidates(train, val)
    # Only now compute the sealed held-out test predictions.
    pred = pd.concat([predictions(val, fitted), predictions(test, fitted)], ignore_index=True)
    test_predictions = pred.loc[pred.split == "test"].copy()
    test_predictions = test_predictions.merge(test[["day", "segment", "humidity_pct"]],
                                              on=["day", "segment"], validate="many_to_one")
    # Humidity bin edges are learned from train only. Duplicated edges signal unusable bins.
    edges = np.quantile(train.humidity_pct, [0.25, 0.5, 0.75])
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Degenerate train-only humidity bin cutoffs")
    test_predictions["humidity_train_quartile"] = 1 + np.searchsorted(
        edges, test_predictions.humidity_pct.to_numpy(), side="right")
    args.out.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out / "segment_features.csv", index=False, lineterminator="\n")
    manifest.to_csv(args.out / "split_manifest.csv", index=False, lineterminator="\n")
    daily_v1(features).to_csv(args.out / "v1_daily.csv", index=False, lineterminator="\n")
    selection.to_csv(args.out / "validation_selection.csv", index=False, lineterminator="\n")
    test_predictions.to_csv(args.out / "test_predictions.csv", index=False, lineterminator="\n")
    overall = aggregate_error(test_predictions, ["model"])
    overall.to_csv(args.out / "test_summary.csv", index=False, lineterminator="\n")
    for filename, keys in (("test_by_day.csv", ["model", "day"]),
                           ("test_by_position.csv", ["model", "segment"]),
                           ("test_by_co_level.csv", ["model", "co_reference_ppm"]),
                           ("test_by_humidity.csv", ["model", "humidity_train_quartile"])):
        aggregate_error(test_predictions, keys).to_csv(args.out / filename, index=False,
                                                         lineterminator="\n")
    summary = {
        "run_status": "completed_lab_calibration_v3_holdout_once", "dataset": "UCI487 laboratory",
        "source_archive_sha256": sha256(args.archive), "script_sha256": sha256(Path(__file__)),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "seed": SEED, "alphas": ALPHAS,
        "settled_window_s": 300, "day_split": "chronological 8/2/3",
        "position_split_each_co_level": "6/2/2, identical position split for every day",
        "observation_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "unique_test_days": int(test.day.nunique()), "unique_test_positions": int(test.segment.nunique()),
        "train_only_humidity_quartile_cutoffs_pct": edges.tolist(),
        "validation_selection": selection.to_dict(orient="records"),
        "test_summary": overall.to_dict(orient="records"),
        "input_exclusion": "CO reference, flow rate, clock, date and position absent from model input",
        "limitation": "Same CO sequence repeated on 13 days; 60 test observations are 3 days x 20 program positions, not independent field scenarios. Completed segments only, no real-time or accident prediction.",
    }
    (args.out / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
