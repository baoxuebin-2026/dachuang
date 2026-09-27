#!/usr/bin/env python3
"""Reproduce MM256 baseline and run precommitted two-step G1 development ablation."""

from __future__ import annotations

import argparse
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .build_stage9_b2_development_windows import file_sha256, load_development
    from .diagnose_stage9_b2_d1_development import trace_alarms
    from .run_stage9_b2_a_dev_comparison import (
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, ROOT, WINDOWS,
        candidate_thresholds, feature_matrix, naive_seconds, read_csv_gz,
        score_alerts,
    )
else:
    from build_stage9_b2_development_windows import file_sha256, load_development
    from diagnose_stage9_b2_d1_development import trace_alarms
    from run_stage9_b2_a_dev_comparison import (
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, ROOT, WINDOWS,
        candidate_thresholds, feature_matrix, naive_seconds, read_csv_gz,
        score_alerts,
    )


CONFIG = ROOT / "configs/stage9_b2_g1_incremental_v1.json"
PARENT = ROOT / "results/stage9_b2/a_dev_comparison_v1.json"
OUTPUT = ROOT / "results/stage9_b2/g1_incremental_development_v1.json"


def load_auxiliary(path: Path, expected_rows: int, last_date: str) -> dict[str, np.ndarray]:
    """Read only the two same-source channels; stop before selection dates."""
    collected: dict[str, list[np.ndarray]] = {"MM264": [], "AN422": []}
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["methane_data.csv"] or archive.testzip() is not None:
            raise ValueError("Unexpected archive/CRC")
        with archive.open("methane_data.csv") as stream:
            for frame in pd.read_csv(stream, usecols=["year", "month", "day", *collected],
                                     chunksize=150_000):
                dates = pd.to_datetime(frame[["year", "month", "day"]]).dt.strftime("%Y-%m-%d")
                if dates.iloc[0] > last_date:
                    break
                frame = frame.loc[dates <= last_date]
                for name in collected:
                    collected[name].append(frame[name].to_numpy(dtype=np.float32))
    arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
    if any(len(values) != expected_rows for values in arrays.values()):
        raise ValueError("Auxiliary data not aligned to MM256 source rows")
    return arrays


def auxiliary_features(values: np.ndarray, rows: np.ndarray, train_rows: np.ndarray,
                       lower: float, upper: float) -> tuple[np.ndarray, dict]:
    """No sample removal: an invalid channel value gets a train-only median/flag."""
    valid = np.isfinite(values) & (values >= lower) & (values <= upper)
    median = float(np.median(values[train_rows[valid[train_rows]]]))
    now_ok = valid[rows]
    lag_ok = valid[rows - 300]
    now = np.where(now_ok, values[rows], median)
    delta_ok = now_ok & lag_ok
    delta = np.where(delta_ok, values[rows] - values[rows - 300], 0.)
    features = np.column_stack((now, delta, ~now_ok, ~delta_ok)).astype(np.float64)
    return features, {
        "train_only_median": median,
        "current_invalid_anchors": int((~now_ok).sum()),
        "delta_invalid_anchors": int((~delta_ok).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    method_cfg = json.loads(METHOD_CONFIG.read_text(encoding="utf-8"))
    label_cfg = json.loads(LABEL_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(LABEL_REPORT.read_text(encoding="utf-8"))
    if (file_sha256(PARENT) != cfg["parent_baseline_result_sha256"]
            or parent["method_config_sha256"] != file_sha256(METHOD_CONFIG)
            or parent["label_protocol_sha256"] != file_sha256(LABEL_CONFIG)
            or manifest["protocol_sha256"] != file_sha256(LABEL_CONFIG)):
        raise ValueError("Frozen prior manifests differ")
    for path in (WINDOWS, EVENTS):
        saved = next(x for x in manifest["processed_files_ignored_by_git"]
                     if x["path"] == str(path.relative_to(ROOT)))
        if file_sha256(path) != saved["sha256"]:
            raise ValueError(f"Frozen derived input changed: {path}")
    stamps, _, targets = load_development(args.archive, label_cfg)
    auxiliary = load_auxiliary(args.archive, len(stamps), label_cfg["development_dates_inclusive"][1])
    windows, events = read_csv_gz(WINDOWS), read_csv_gz(EVENTS)
    rows_all = windows.source_row.to_numpy(dtype=np.int64)
    labels_all = windows.proxy_label.to_numpy(dtype=np.int8)
    if not np.array_equal(pd.to_datetime(windows.prediction_local_naive)
                          .to_numpy(dtype="datetime64[s]").astype(np.int64), stamps[rows_all]):
        raise ValueError("Window timestamp/source index mismatch")
    cut = naive_seconds(method_cfg["internal_validation_dates_inclusive"][0])
    masks = {
        "train": stamps[rows_all] + method_cfg["split_guard_future_seconds"] < cut,
        "internal_validation": stamps[rows_all] - method_cfg["split_guard_history_seconds"] >= cut,
    }
    event_times = stamps[events.source_row.to_numpy(dtype=np.int64)]
    train_rows = rows_all[masks["train"]]
    medians = {}
    parts = {}
    for part, mask in masks.items():
        rows = rows_all[mask]
        base = feature_matrix(targets["MM256"], rows)
        features = {"MM256_only": base}
        metadata = {}
        for name in ("MM264", "AN422"):
            lo, hi = cfg["auxiliary_validity"][name]
            extra, metadata[name] = auxiliary_features(auxiliary[name], rows, train_rows, lo, hi)
            if name in medians and medians[name] != metadata[name]["train_only_median"]:
                raise ValueError("Training median changed between partitions")
            medians[name] = metadata[name]["train_only_median"]
            variant = "MM256_plus_MM264" if name == "MM264" else "MM256_plus_MM264_plus_AN422"
            features[variant] = np.column_stack((features[list(features)[-1]], extra))
        parts[part] = {
            "rows": rows, "times": stamps[rows], "labels": labels_all[mask],
            "events": event_times[event_times < cut] if part == "train" else event_times[event_times >= cut],
            "features": features, "input_quality": metadata,
        }
        if len(rows) != parent["split"][part]["eligible_windows"]:
            raise ValueError("Baseline denominator changed")
    result = {
        "status": "post_F1_exploratory_development_only; already_viewed_internal_validation; no external generalization",
        "protocol_sha256": file_sha256(CONFIG), "source_archive_sha256": file_sha256(args.archive),
        "parent_baseline_result_sha256": file_sha256(PARENT),
        "derived_input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in (WINDOWS, EVENTS)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "partitions": {p: {"eligible_windows": len(x["rows"]), "proxy_events": len(x["events"]),
                           "input_quality": x["input_quality"]} for p, x in parts.items()},
        "variants": {},
    }
    for variant in cfg["variants"]:
        train, valid = parts["train"], parts["internal_validation"]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=20260926))
        model.fit(train["features"][variant], train["labels"])
        if variant == "MM256_only" and not np.allclose(
                model[-1].coef_[0], parent["model_coefficients_scaled"]["coefficients"],
                rtol=0, atol=1e-11):
            raise ValueError("Parent baseline coefficients not reproduced")
        train_scores = model.predict_proba(train["features"][variant])[:, 1]
        valid_scores = model.predict_proba(valid["features"][variant])[:, 1]
        thresholds = candidate_thresholds(train_scores)
        train_records = [score_alerts(train["rows"], train["times"], train["labels"],
                                      np.arange(len(train["events"])), train["events"],
                                      train_scores, float(threshold))
                         for threshold in thresholds]
        comparison = {}
        for budget in (cfg["primary_training_false_new_alert_budget_per_24_background_hours"],
                       cfg["secondary_training_budget"]):
            allowed = [i for i, rec in enumerate(train_records)
                       if rec["false_per_24_background_hours"] <= budget]
            best = min(allowed, key=lambda i: (
                -train_records[i]["matched_events"],
                train_records[i]["false_per_24_background_hours"],
                train_records[i]["duplicate_or_unmatched_positive_new_alerts"],
                -thresholds[i],
            ))
            threshold = float(thresholds[best])
            traces = {}
            for part, scores in (("train", train_scores), ("internal_validation", valid_scores)):
                item = parts[part]
                traced = trace_alarms(item["times"], item["labels"], item["events"],
                                      scores, threshold, [60, 180])
                frozen = score_alerts(item["rows"], item["times"], item["labels"],
                                      np.arange(len(item["events"])), item["events"],
                                      scores, threshold)
                if (traced["matched_events_any_lead"] != frozen["matched_events"]
                        or traced["false_new_alerts_background"] != frozen["false_new_alerts_background"]):
                    raise ValueError("Diagnostic/frozen scoring disagree")
                traced["false_per_24_background_hours"] = frozen["false_per_24_background_hours"]
                traces[part] = traced
            if variant == "MM256_only":
                expected = parent["methods"]["simple_logistic"]["budget_comparisons"][str(budget)]
                if (expected["threshold_selected_from_train"] != threshold
                        or any(expected[p]["matched_events"] != traces[p]["matched_events_any_lead"]
                               or expected[p]["false_new_alerts_background"] != traces[p]["false_new_alerts_background"]
                               for p in traces)):
                    raise ValueError("Frozen baseline counts/threshold not reproduced")
            comparison[str(budget)] = {"training_threshold": threshold, "traces": traces}
        result["variants"][variant] = {
            "feature_count": int(train["features"][variant].shape[1]),
            "threshold_candidates": int(len(thresholds)),
            "coefficients_scaled": list(map(float, model[-1].coef_[0])),
            "budget_comparisons": comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({v: {b: {
        "val_any": r["traces"]["internal_validation"]["matched_events_any_lead"],
        "val_60s": r["traces"]["internal_validation"]["matched_events_at_least_lead_seconds"]["60"],
        "val_false": r["traces"]["internal_validation"]["false_new_alerts_background"],
    } for b, r in x["budget_comparisons"].items()} for v, x in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
