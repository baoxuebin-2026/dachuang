#!/usr/bin/env python3
"""Frozen UCI487 leave-one-replica-out high-to-low CO stress test."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EXPECTED_SHA = "cb5ce4a6af1a51b933d1979952d7845f0e9baac54e15a81a1e3599e8b85905d4"
SENSORS = [f"R{i} (MOhm)" for i in range(8, 15)]
FACTORS = (0.5, 0.75, 1.0)
TRAIN_POSITIVE = (6.67, 8.89, 11.11, 13.33, 15.56, 17.78, 20.0)
TEST_POSITIVE = (2.22, 4.44)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phase_medians(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    part = frame.loc[mask]
    high = part["Heater voltage (V)"].to_numpy() > 0.55
    if high.sum() < 20 or (~high).sum() < 20:
        raise ValueError("Insufficient samples in a heater phase")
    values = np.log(part[SENSORS].to_numpy(dtype=float))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite log resistance")
    return np.stack((np.median(values[~high], axis=0), np.median(values[high], axis=0)), axis=1)


def extract(archive: Path) -> pd.DataFrame:
    rows = []
    usecols = ["Time (s)", "CO (ppm)", "Humidity (%r.h.)", "Heater voltage (V)", *SENSORS]
    with zipfile.ZipFile(archive) as outer:
        raw = outer.read("gas-sensor-array-temperature-modulation.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
        names = sorted(n for n in inner.namelist() if n.endswith(".csv"))
        if len(names) != 13:
            raise ValueError("Expected 13 days")
        for day_index, name in enumerate(names):
            frame = pd.read_csv(inner.open(name), usecols=usecols)
            time = frame["Time (s)"].to_numpy()
            baseline = phase_medians(frame, (time >= 600.0) & (time < 900.0))
            segment = np.floor((time - 900.0) / 900.0).astype(int)
            within = (time - 900.0) - segment * 900.0
            for condition in range(100):
                mask = (segment == condition) & (within >= 600.0) & (within < 900.0)
                response = phase_medians(frame, mask) - baseline
                label_rows = frame.loc[mask, ["CO (ppm)", "Humidity (%r.h.)"]]
                co = float(label_rows["CO (ppm)"].median())
                humidity = float(label_rows["Humidity (%r.h.)"].median())
                for device_index, sensor in enumerate(SENSORS):
                    rows.append({
                        "day": Path(name).stem,
                        "day_index": day_index,
                        "condition": condition,
                        "device": sensor.split()[0],
                        "co_ppm": co,
                        "humidity": humidity,
                        "response_low": float(response[device_index, 0]),
                        "response_high": float(response[device_index, 1]),
                    })
    table = pd.DataFrame(rows)
    if len(table) != 13 * 100 * 7 or table.isna().any().any():
        raise ValueError("Unexpected extracted table")
    return table


def rms(x: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(x * x, axis=1, keepdims=True))
    return x / np.maximum(scale, 1e-12)


def model(name: str):
    if name == "logistic_amplitude":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=2026))
    return make_pipeline(StandardScaler(), SVC(
        C=1.0, gamma="scale", class_weight="balanced"))


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "co_recall": float(recall_score(y, pred, zero_division=0)),
        "co_precision": float(precision_score(y, pred, zero_division=0)),
        "false_negatives": int(np.sum((y == 1) & (pred == 0))),
        "false_positives": int(np.sum((y == 0) & (pred == 1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.archive) != EXPECTED_SHA:
        raise ValueError("Source archive hash mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_cross_device_fit_or_score":
        raise ValueError("Protocol is not frozen")
    data = extract(args.archive)
    features = ["response_low", "response_high"]
    predictions = []
    blocks = []
    for held in sorted(data.device.unique()):
        source = data[(data.device != held) & (data.day_index < 10) &
                      (data.co_ppm.isin((0.0, *TRAIN_POSITIVE)))].copy()
        target = data[(data.device == held) & (data.day_index >= 10) &
                      (data.co_ppm.isin((0.0, *TEST_POSITIVE)))].copy()
        x_source = source[features].to_numpy()
        y_source = (source.co_ppm > 0).astype(int).to_numpy()
        x_target = target[features].to_numpy()
        y_target = (target.co_ppm > 0).astype(int).to_numpy()
        for method in ("svm_raw", "svm_amplitude", "svm_rms", "logistic_amplitude"):
            if method in ("svm_amplitude", "logistic_amplitude"):
                x_fit = np.concatenate([x_source * factor for factor in FACTORS])
                y_fit = np.tile(y_source, len(FACTORS))
                x_eval = x_target
            elif method == "svm_rms":
                x_fit, y_fit, x_eval = rms(x_source), y_source, rms(x_target)
            else:
                x_fit, y_fit, x_eval = x_source, y_source, x_target
            estimator = model(method)
            estimator.fit(x_fit, y_fit)
            pred = estimator.predict(x_eval)
            record = target[["day", "day_index", "condition", "device", "co_ppm", "humidity"]].copy()
            record["method"] = method
            record["truth"] = y_target
            record["prediction"] = pred
            predictions.append(record)
            for day, part in record.groupby("day", sort=True):
                metric = score(part.truth.to_numpy(), part.prediction.to_numpy())
                blocks.append({"held_device": held, "day": day, "method": method, **metric})
    pred = pd.concat(predictions, ignore_index=True)
    block = pd.DataFrame(blocks)
    summary = block.groupby("method", sort=True).agg(
        blocks=("macro_f1", "size"),
        mean_macro_f1=("macro_f1", "mean"),
        median_macro_f1=("macro_f1", "median"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_co_recall=("co_recall", "mean"),
        mean_co_precision=("co_precision", "mean"),
    ).reset_index()
    pivot = block.pivot(index=["held_device", "day"], columns="method", values="macro_f1")
    gain = pivot["svm_amplitude"] - pivot["svm_raw"]
    amp = summary.set_index("method").loc["svm_amplitude"]
    strongest = float(summary.mean_macro_f1.max())
    per_device = block[block.method == "svm_amplitude"].groupby("held_device").macro_f1.mean()
    checks = {
        "mean_macro_f1_ge_0_80": bool(amp.mean_macro_f1 >= 0.80),
        "mean_co_recall_ge_0_80": bool(amp.mean_co_recall >= 0.80),
        "positive_blocks_ge_15": bool((gain > 0).sum() >= 15),
        "median_gain_ge_0_05": bool(gain.median() >= 0.05),
        "within_0_03_of_strongest": bool(float(amp.mean_macro_f1) >= strongest - 0.03),
        "all_devices_mean_f1_ge_0_70": bool((per_device >= 0.70).all()),
    }
    result = {
        "protocol": str(args.protocol),
        "source_sha256": EXPECTED_SHA,
        "rows": int(len(data)),
        "train_days": sorted(data.loc[data.day_index < 10, "day"].unique()),
        "test_days": sorted(data.loc[data.day_index >= 10, "day"].unique()),
        "paired_gain": {
            "positive_blocks": int((gain > 0).sum()),
            "zero_blocks": int((gain == 0).sum()),
            "negative_blocks": int((gain < 0).sum()),
            "median_macro_f1_gain": float(gain.median()),
            "mean_macro_f1_gain": float(gain.mean()),
        },
        "checks": checks,
        "passes_frozen_success_rule": bool(all(checks.values())),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out / "features.csv", index=False)
    pred.to_csv(args.out / "predictions.csv", index=False)
    block.to_csv(args.out / "block_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(result["paired_gain"], indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
