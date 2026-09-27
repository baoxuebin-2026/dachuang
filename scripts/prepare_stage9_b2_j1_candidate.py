#!/usr/bin/env python3
"""Fit J1 only on pre-May development data; save a reviewable selection candidate."""

from __future__ import annotations

import argparse
import json
import platform
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
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, OUTPUT as A_RESULT,
        ROOT, WINDOWS, candidate_thresholds, feature_matrix, naive_seconds,
        read_csv_gz, score_alerts,
    )
    from .run_stage9_b2_g1_incremental import auxiliary_features, load_auxiliary
else:
    from build_stage9_b2_development_windows import file_sha256, load_development
    from diagnose_stage9_b2_d1_development import trace_alarms
    from run_stage9_b2_a_dev_comparison import (
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, OUTPUT as A_RESULT,
        ROOT, WINDOWS, candidate_thresholds, feature_matrix, naive_seconds,
        read_csv_gz, score_alerts,
    )
    from run_stage9_b2_g1_incremental import auxiliary_features, load_auxiliary


CONFIG = ROOT / "configs/stage9_b2_j1_robust_candidate_v1.json"
G1_RESULT = ROOT / "results/stage9_b2/g1_incremental_development_v1.json"
H1_RESULT = ROOT / "results/stage9_b2/h1_extreme_sensitivity_v1.json"
OUTPUT = ROOT / "results/stage9_b2/j1_robust_candidate_development_v1.json"


def score_from_saved_model(features: np.ndarray, saved: dict) -> np.ndarray:
    """Stable standardization/logistic inference shared with later selection code."""
    z = (features - np.asarray(saved["scaler_mean"])) / np.asarray(saved["scaler_scale"])
    logit = z @ np.asarray(saved["coefficients_scaled"]) + saved["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -700., 700.)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    g1 = json.loads(G1_RESULT.read_text(encoding="utf-8"))
    method = json.loads(METHOD_CONFIG.read_text(encoding="utf-8"))
    label = json.loads(LABEL_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(LABEL_REPORT.read_text(encoding="utf-8"))
    original_a = json.loads(A_RESULT.read_text(encoding="utf-8"))
    if (file_sha256(args.archive) != cfg["source_sha256"]
            or file_sha256(G1_RESULT) != cfg["g1_result_sha256"]
            or file_sha256(H1_RESULT) != cfg["h1_result_sha256"]
            or manifest["protocol_sha256"] != file_sha256(LABEL_CONFIG)
            or original_a["method_config_sha256"] != file_sha256(METHOD_CONFIG)):
        raise ValueError("Frozen source/protocol/result mismatch")
    for path in (WINDOWS, EVENTS):
        expected = next(x for x in manifest["processed_files_ignored_by_git"]
                        if x["path"] == str(path.relative_to(ROOT)))
        if file_sha256(path) != expected["sha256"]:
            raise ValueError("Derived input mismatch")
    stamps, _, targets = load_development(args.archive, label)
    aux = load_auxiliary(args.archive, len(stamps), label["development_dates_inclusive"][1])
    windows, events = read_csv_gz(WINDOWS), read_csv_gz(EVENTS)
    rows_all = windows.source_row.to_numpy(dtype=np.int64)
    labels_all = windows.proxy_label.to_numpy(dtype=np.int8)
    if not np.array_equal(pd.to_datetime(windows.prediction_local_naive)
                          .to_numpy(dtype="datetime64[s]").astype(np.int64), stamps[rows_all]):
        raise ValueError("Window timestamps differ from source")
    event_times = stamps[events.source_row.to_numpy(dtype=np.int64)]
    cut = naive_seconds(method["internal_validation_dates_inclusive"][0])
    masks = {
        "train": stamps[rows_all] + method["split_guard_future_seconds"] < cut,
        "already_seen_development_check": stamps[rows_all] - method["split_guard_history_seconds"] >= cut,
    }
    train_rows = rows_all[masks["train"]]
    parts = {}
    median = None
    for part, mask in masks.items():
        rows = rows_all[mask]
        base = feature_matrix(targets["MM256"], rows)
        mm264, quality = auxiliary_features(aux["MM264"], rows, train_rows,
                                             cfg["mm264_valid_min_inclusive"],
                                             cfg["mm264_valid_max_inclusive_pct_ch4"])
        if median is not None and median != quality["train_only_median"]:
            raise ValueError("Training-only median changed")
        median = quality["train_only_median"]
        parts[part] = {
            "rows": rows, "times": stamps[rows], "labels": labels_all[mask],
            "events": event_times[event_times < cut] if part == "train" else event_times[event_times >= cut],
            "input_quality": quality,
            "features": {
                "original_MM256_only_baseline": base,
                "robust_MM256_plus_MM264": np.column_stack((base, mm264)),
            },
        }
        g1_part = "train" if part == "train" else "internal_validation"
        if len(rows) != g1["partitions"][g1_part]["eligible_windows"]:
            raise ValueError("Development denominator differs from G1")
    train, check = parts["train"], parts["already_seen_development_check"]
    result = {
        "status": "development_only_candidate; selection_period_not_read_or_scored; validation_already_inspected",
        "protocol_sha256": file_sha256(CONFIG), "source_archive_sha256": file_sha256(args.archive),
        "label_protocol_sha256": file_sha256(LABEL_CONFIG),
        "method_protocol_sha256": file_sha256(METHOD_CONFIG),
        "g1_result_sha256": file_sha256(G1_RESULT), "h1_result_sha256": file_sha256(H1_RESULT),
        "derived_files_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in (WINDOWS, EVENTS)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "train_only_mm264_median": median,
        "partitions": {name: {"eligible_windows": int(len(p["rows"])),
                              "proxy_events": int(len(p["events"])),
                              "input_quality": p["input_quality"]} for name, p in parts.items()},
        "models": {},
    }
    for name in cfg["variants"]:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=20260926))
        model.fit(train["features"][name], train["labels"])
        saved = {
            "feature_count": train["features"][name].shape[1],
            "scaler_mean": list(map(float, model[0].mean_)),
            "scaler_scale": list(map(float, model[0].scale_)),
            "coefficients_scaled": list(map(float, model[-1].coef_[0])),
            "intercept": float(model[-1].intercept_[0]),
        }
        train_scores = model.predict_proba(train["features"][name])[:, 1]
        check_scores = model.predict_proba(check["features"][name])[:, 1]
        for features, scores in ((train["features"][name], train_scores),
                                 (check["features"][name], check_scores)):
            if not np.allclose(score_from_saved_model(features, saved), scores, atol=1e-12, rtol=0):
                raise ValueError("Serialized scaler/logistic inference disagrees")
        thresholds = candidate_thresholds(train_scores)
        records = [score_alerts(train["rows"], train["times"], train["labels"],
                                np.arange(len(train["events"])), train["events"],
                                train_scores, float(threshold)) for threshold in thresholds]
        budget = 2
        allowed = [i for i, rec in enumerate(records)
                   if rec["false_per_24_background_hours"] <= budget]
        best = min(allowed, key=lambda i: (
            -records[i]["matched_events"], records[i]["false_per_24_background_hours"],
            records[i]["duplicate_or_unmatched_positive_new_alerts"], -thresholds[i],
        ))
        threshold = float(thresholds[best])
        if name == "original_MM256_only_baseline":
            parent = original_a["methods"]["simple_logistic"]["budget_comparisons"]["2"]
            if (not np.allclose(saved["coefficients_scaled"],
                                original_a["model_coefficients_scaled"]["coefficients"],
                                atol=1e-11, rtol=0)
                    or threshold != parent["threshold_selected_from_train"]):
                raise ValueError("MM256 baseline failed to reproduce frozen A")
        saved["training_threshold_budget_2"] = threshold
        saved["training_threshold_candidates"] = len(thresholds)
        saved["train_trace"] = trace_alarms(train["times"], train["labels"], train["events"],
                                             train_scores, threshold, [60, 180])
        saved["already_seen_development_trace"] = trace_alarms(
            check["times"], check["labels"], check["events"], check_scores, threshold,
            [60, 180])
        result["models"][name] = saved
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"threshold": m["training_threshold_budget_2"],
                             "train_60s": m["train_trace"]["matched_events_at_least_lead_seconds"]["60"],
                             "seen_dev_60s": m["already_seen_development_trace"]["matched_events_at_least_lead_seconds"]["60"],
                             "seen_dev_false": m["already_seen_development_trace"]["false_new_alerts_background"]}
                      for name, m in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
