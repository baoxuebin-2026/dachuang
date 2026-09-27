#!/usr/bin/env python3
"""H1: frozen G1 models/thresholds, input-only MM264 >10% stress replay."""

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
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, ROOT, WINDOWS,
        feature_matrix, naive_seconds, read_csv_gz, score_alerts,
    )
    from .run_stage9_b2_g1_incremental import auxiliary_features, load_auxiliary
else:
    from build_stage9_b2_development_windows import file_sha256, load_development
    from diagnose_stage9_b2_d1_development import trace_alarms
    from run_stage9_b2_a_dev_comparison import (
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, ROOT, WINDOWS,
        feature_matrix, naive_seconds, read_csv_gz, score_alerts,
    )
    from run_stage9_b2_g1_incremental import auxiliary_features, load_auxiliary


CONFIG = ROOT / "configs/stage9_b2_h1_extreme_sensitivity_v1.json"
G1_CONFIG = ROOT / "configs/stage9_b2_g1_incremental_v1.json"
G1_RESULT = ROOT / "results/stage9_b2/g1_incremental_development_v1.json"
OUTPUT = ROOT / "results/stage9_b2/h1_extreme_sensitivity_v1.json"


def stressed_mm264(values: np.ndarray, rows: np.ndarray, median: float,
                   cutoff: float) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(values) & (values >= 0) & (values <= cutoff)
    now_ok, past_ok = valid[rows], valid[rows - 300]
    both = now_ok & past_ok
    features = np.column_stack((
        np.where(now_ok, values[rows], median),
        np.where(both, values[rows] - values[rows - 300], 0.),
        ~now_ok, ~both,
    )).astype(np.float64)
    extreme_current = values[rows] > cutoff
    extreme_lag = values[rows - 300] > cutoff
    return features, {
        "new_extreme_current_anchors": int(extreme_current.sum()),
        "new_extreme_lag_anchors": int(extreme_lag.sum()),
        "distinct_anchors_with_either_extreme": int((extreme_current | extreme_lag).sum()),
        "first_affected_local_naive": None,
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
    g1 = json.loads(G1_RESULT.read_text(encoding="utf-8"))
    g1_cfg = json.loads(G1_CONFIG.read_text(encoding="utf-8"))
    method_cfg = json.loads(METHOD_CONFIG.read_text(encoding="utf-8"))
    label_cfg = json.loads(LABEL_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(LABEL_REPORT.read_text(encoding="utf-8"))
    if (file_sha256(G1_RESULT) != cfg["parent_result_sha256"]
            or g1["protocol_sha256"] != file_sha256(G1_CONFIG)
            or g1["source_archive_sha256"] != file_sha256(args.archive)
            or manifest["protocol_sha256"] != file_sha256(LABEL_CONFIG)):
        raise ValueError("Frozen parent/source/protocol differ")
    for path in (WINDOWS, EVENTS):
        saved = next(x for x in manifest["processed_files_ignored_by_git"]
                     if x["path"] == str(path.relative_to(ROOT)))
        if file_sha256(path) != saved["sha256"]:
            raise ValueError("Frozen derived input changed")
    stamps, _, targets = load_development(args.archive, label_cfg)
    aux = load_auxiliary(args.archive, len(stamps), label_cfg["development_dates_inclusive"][1])
    windows, events = read_csv_gz(WINDOWS), read_csv_gz(EVENTS)
    all_rows = windows.source_row.to_numpy(dtype=np.int64)
    all_labels = windows.proxy_label.to_numpy(dtype=np.int8)
    if not np.array_equal(pd.to_datetime(windows.prediction_local_naive)
                          .to_numpy(dtype="datetime64[s]").astype(np.int64), stamps[all_rows]):
        raise ValueError("Frozen source row timestamps changed")
    cut = naive_seconds(method_cfg["internal_validation_dates_inclusive"][0])
    train_mask = stamps[all_rows] + method_cfg["split_guard_future_seconds"] < cut
    val_mask = stamps[all_rows] - method_cfg["split_guard_history_seconds"] >= cut
    event_times = stamps[events.source_row.to_numpy(dtype=np.int64)]
    train_rows, val_rows = all_rows[train_mask], all_rows[val_mask]
    median = g1["partitions"]["train"]["input_quality"]["MM264"]["train_only_median"]
    original = {}
    for name, rows in (("train", train_rows), ("internal_validation", val_rows)):
        base = feature_matrix(targets["MM256"], rows)
        mm264, original_meta = auxiliary_features(aux["MM264"], rows, train_rows, 0, 30)
        wind, wind_meta = auxiliary_features(aux["AN422"], rows, train_rows, -5, 5)
        expected = g1["partitions"][name]["input_quality"]
        if original_meta != expected["MM264"] or wind_meta != expected["AN422"]:
            raise ValueError("Parent preprocessing not reproduced")
        original[name] = {
            "MM256_plus_MM264": np.column_stack((base, mm264)),
            "MM256_plus_MM264_plus_AN422": np.column_stack((base, mm264, wind)),
        }
    mm264_stress, change = stressed_mm264(aux["MM264"], val_rows, median,
                                           cfg["posthoc_extreme_cutoff_pct_ch4"])
    affected = (aux["MM264"][val_rows] > cfg["posthoc_extreme_cutoff_pct_ch4"]) | (
        aux["MM264"][val_rows - 300] > cfg["posthoc_extreme_cutoff_pct_ch4"])
    if affected.any():
        change["first_affected_local_naive"] = str(np.datetime64(int(stamps[val_rows[affected][0]]), "s"))
    stressed = {
        "MM256_plus_MM264": np.column_stack((original["internal_validation"]["MM256_plus_MM264"][:, :5], mm264_stress)),
        "MM256_plus_MM264_plus_AN422": np.column_stack((
            original["internal_validation"]["MM256_plus_MM264_plus_AN422"][:, :5],
            mm264_stress,
            original["internal_validation"]["MM256_plus_MM264_plus_AN422"][:, 9:],
        )),
    }
    result = {
        "status": "posthoc_fixed_model_fixed_threshold_input_stress_development_only",
        "protocol_sha256": file_sha256(CONFIG),
        "parent_result_sha256": file_sha256(G1_RESULT),
        "source_archive_sha256": file_sha256(args.archive),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "affected_internal_validation_anchors": change,
        "same_eligible_windows": int(len(val_rows)),
        "variants": {},
    }
    val_times, val_labels = stamps[val_rows], all_labels[val_mask]
    val_events = event_times[event_times >= cut]
    for variant in stressed:
        train_features = original["train"][variant]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=20260926))
        model.fit(train_features, all_labels[train_mask])
        frozen = g1["variants"][variant]
        if not np.allclose(model[-1].coef_[0], frozen["coefficients_scaled"], rtol=0, atol=1e-11):
            raise ValueError("G1 original model not reproduced")
        original_scores = model.predict_proba(original["internal_validation"][variant])[:, 1]
        stressed_scores = model.predict_proba(stressed[variant])[:, 1]
        if np.any((~affected) & (original_scores != stressed_scores)):
            raise ValueError("Stress unexpectedly changed unaffected windows")
        comparisons = {}
        for budget, original_run in frozen["budget_comparisons"].items():
            threshold = float(original_run["training_threshold"])
            before = trace_alarms(val_times, val_labels, val_events, original_scores, threshold, [60, 180])
            for key in ("matched_events_any_lead", "matched_events_at_least_lead_seconds",
                        "false_new_alerts_background", "event_records"):
                if before[key] != original_run["traces"]["internal_validation"][key]:
                    raise ValueError(f"G1 original result changed: {variant} {budget} {key}")
            after = trace_alarms(val_times, val_labels, val_events, stressed_scores, threshold, [60, 180])
            reference = score_alerts(val_rows, val_times, val_labels, np.arange(len(val_events)),
                                     val_events, stressed_scores, threshold)
            if (after["matched_events_any_lead"] != reference["matched_events"]
                    or after["false_new_alerts_background"] != reference["false_new_alerts_background"]):
                raise ValueError("Frozen scoring and H1 diagnostic disagree")
            after["false_per_24_background_hours"] = reference["false_per_24_background_hours"]
            before["false_per_24_background_hours"] = original_run["traces"]["internal_validation"][
                "false_per_24_background_hours"]
            comparisons[budget] = {
                "unchanged_training_threshold": threshold,
                "original": before,
                "stressed": after,
                "original_matched_event_onsets": [x["onset_local_naive"] for x in before["event_records"]],
                "stressed_matched_event_onsets": [x["onset_local_naive"] for x in after["event_records"]],
            }
        result["variants"][variant] = {
            "score_changed_anchors": int(np.count_nonzero(original_scores != stressed_scores)),
            "budget_comparisons": comparisons,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {budget: {
        mode: {"any": record["matched_events_any_lead"],
               "lead_60s": record["matched_events_at_least_lead_seconds"]["60"],
               "false": record["false_new_alerts_background"]}
        for mode, record in (("original", item["original"]), ("stressed", item["stressed"]))}
        for budget, item in v["budget_comparisons"].items()}
        for name, v in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
