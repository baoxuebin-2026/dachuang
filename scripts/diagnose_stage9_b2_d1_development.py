#!/usr/bin/env python3
"""Replay frozen B2-A alarms for development-only lead and gate diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .build_stage9_b2_development_windows import file_sha256, load_development
    from .run_stage9_b2_a_dev_comparison import (
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, OUTPUT as PARENT_OUTPUT,
        ROOT, WINDOWS, feature_matrix, naive_seconds, read_csv_gz, score_alerts,
    )
else:
    from build_stage9_b2_development_windows import file_sha256, load_development
    from run_stage9_b2_a_dev_comparison import (
        EVENTS, LABEL_CONFIG, LABEL_REPORT, METHOD_CONFIG, OUTPUT as PARENT_OUTPUT,
        ROOT, WINDOWS, feature_matrix, naive_seconds, read_csv_gz, score_alerts,
    )


CONFIG = ROOT / "configs/stage9_b2_d1_diagnostic_v1.json"
OUTPUT = ROOT / "results/stage9_b2/d1_development_diagnostic_v1.json"


def trace_alarms(times: np.ndarray, labels: np.ndarray, events: np.ndarray,
                 scores: np.ndarray, threshold: float, lead_cutoffs: list[int]) -> dict:
    """Mirror frozen one-alarm-to-one-future-event scoring, preserving dates."""
    evaluable = (np.searchsorted(times, events) > np.searchsorted(times, events - 360))
    active = scores >= threshold
    preceding_adjacent_active = np.r_[False, active[:-1] & (np.diff(times) == 60)]
    new = np.flatnonzero(active & ~preceding_adjacent_active)
    matched = np.zeros(len(events), dtype=bool)
    event_records = []
    false_by_date: Counter[str] = Counter()
    new_reason: Counter[str] = Counter()
    false_reason: Counter[str] = Counter()
    duplicate_positive = 0
    for k in new:
        reason = ("first_partition_anchor" if k == 0 else
                  "nonadjacent_eligible_anchor" if times[k] - times[k - 1] != 60 else
                  "score_crossing")
        new_reason[reason] += 1
        j = np.searchsorted(events, times[k], side="right")
        while j < len(events) and events[j] <= times[k] + 360 and (matched[j] or not evaluable[j]):
            j += 1
        if j < len(events) and events[j] <= times[k] + 360 and evaluable[j]:
            matched[j] = True
            event_records.append({
                "onset_local_naive": str(np.datetime64(int(events[j]), "s")),
                "alarm_local_naive": str(np.datetime64(int(times[k]), "s")),
                "lead_seconds": int(events[j] - times[k]),
                "new_alert_reason": reason,
            })
        elif labels[k] == 0:
            day = str(np.datetime64(int(times[k]), "s"))[:10]
            false_by_date[day] += 1
            false_reason[reason] += 1
        else:
            duplicate_positive += 1
    lead = [record["lead_seconds"] for record in event_records]
    return {
        "evaluable_events": int(evaluable.sum()),
        "matched_events_any_lead": len(event_records),
        "matched_events_at_least_lead_seconds": {
            str(cutoff): sum(v >= cutoff for v in lead) for cutoff in lead_cutoffs
        },
        "new_alert_episodes": len(new),
        "false_new_alerts_background": int(sum(false_by_date.values())),
        "duplicate_or_unmatched_positive_new_alerts": duplicate_positive,
        "new_alert_reason": dict(new_reason),
        "false_alarm_reason": dict(false_reason),
        "background_false_by_date": dict(sorted(false_by_date.items())),
        "event_records": event_records,
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
    prior = json.loads(PARENT_OUTPUT.read_text(encoding="utf-8"))
    assert file_sha256(PARENT_OUTPUT) == cfg["parent_result_sha256"]
    assert prior["method_config_sha256"] == file_sha256(METHOD_CONFIG)
    method_cfg = json.loads(METHOD_CONFIG.read_text(encoding="utf-8"))
    label_cfg = json.loads(LABEL_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(LABEL_REPORT.read_text(encoding="utf-8"))
    for p in (WINDOWS, EVENTS):
        saved = next(e for e in manifest["processed_files_ignored_by_git"]
                     if e["path"] == str(p.relative_to(ROOT)))
        assert saved["sha256"] == file_sha256(p)
    windows = read_csv_gz(WINDOWS)
    events = read_csv_gz(EVENTS)
    stamps, _, target = load_development(args.archive, label_cfg)
    x = target[cfg["source_target"]]
    rows = windows.source_row.to_numpy(dtype=np.int64)
    y = windows.proxy_label.to_numpy(dtype=np.int8)
    assert np.array_equal(pd.to_datetime(windows.prediction_local_naive)
                          .to_numpy(dtype="datetime64[s]").astype(np.int64), stamps[rows])
    cut = naive_seconds(method_cfg["internal_validation_dates_inclusive"][0])
    mask = {
        "train": stamps[rows] + method_cfg["split_guard_future_seconds"] < cut,
        "internal_validation": stamps[rows] - method_cfg["split_guard_history_seconds"] >= cut,
    }
    event_times = stamps[events.source_row.to_numpy(dtype=np.int64)]
    parts = {}
    for name, select in mask.items():
        selected = rows[select]
        parts[name] = {
            "rows": selected, "times": stamps[selected], "labels": y[select],
            "features": feature_matrix(x, selected),
            "events": event_times[event_times < cut] if name == "train" else event_times[event_times >= cut],
        }
        assert prior["split"][name]["eligible_windows"] == len(selected)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=20260926))
    model.fit(parts["train"]["features"], parts["train"]["labels"])
    assert np.allclose(model[-1].coef_[0], prior["model_coefficients_scaled"]["coefficients"],
                       rtol=0, atol=1e-11)
    score_by_part = {}
    for name, part in parts.items():
        f = part["features"]
        score_by_part[name] = {
            "current_concentration": f[:, 0],
            "causal_slope_extrapolation": f[:, 0] + 360 * f[:, 2] / 300,
            "simple_logistic": model.predict_proba(f)[:, 1],
            "gate_only_null": np.ones(len(f)),
        }
    result = {
        "status": "posthoc_development_only; original thresholds and any-lead outcomes unchanged",
        "config_sha256": file_sha256(CONFIG),
        "parent_result_sha256": file_sha256(PARENT_OUTPUT),
        "source_archive_sha256": label_cfg["source_sha256"],
        "lead_cutoffs_seconds": cfg["additional_lead_time_cutoffs_seconds"],
        "traces": {},
    }
    requested = [tuple(pair) for pair in cfg["diagnosed_methods_and_training_budgets"]]
    for method, budget in [*requested, ("gate_only_null", "not_selected")]:
        key = f"{method}|{budget}"
        result["traces"][key] = {}
        if method == "gate_only_null":
            threshold = .5
        else:
            threshold = prior["methods"][method]["budget_comparisons"][str(budget)][
                "threshold_selected_from_train"]
            threshold = np.inf if threshold == "always_silent" else float(threshold)
        for name, part in parts.items():
            s = score_by_part[name][method]
            traced = trace_alarms(part["times"], part["labels"], part["events"], s,
                                  threshold, cfg["additional_lead_time_cutoffs_seconds"])
            frozen = score_alerts(part["rows"], part["times"], part["labels"],
                                  np.arange(len(part["events"])), part["events"], s, threshold)
            for field, from_frozen in (
                ("evaluable_events", "evaluable_events"),
                ("matched_events_any_lead", "matched_events"),
                ("new_alert_episodes", "new_alert_episodes"),
                ("false_new_alerts_background", "false_new_alerts_background"),
                ("duplicate_or_unmatched_positive_new_alerts",
                 "duplicate_or_unmatched_positive_new_alerts"),
            ):
                assert traced[field] == frozen[from_frozen], (key, name, field)
            if method != "gate_only_null":
                published = prior["methods"][method]["budget_comparisons"][str(budget)][name]
                for field, published_field in (
                    ("evaluable_events", "evaluable_events"),
                    ("matched_events_any_lead", "matched_events"),
                    ("new_alert_episodes", "new_alert_episodes"),
                    ("false_new_alerts_background", "false_new_alerts_background"),
                ):
                    assert traced[field] == published[published_field], (key, name, field)
            result["traces"][key][name] = traced
    gate = result["traces"]["gate_only_null|not_selected"]
    current = result["traces"]["current_concentration|4"]
    for name in parts:
        assert gate[name]["matched_events_any_lead"] == current[name]["matched_events_any_lead"]
        assert gate[name]["false_new_alerts_background"] == current[name]["false_new_alerts_background"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {
        part: {"any": trace["matched_events_any_lead"],
               "lead": trace["matched_events_at_least_lead_seconds"],
               "false": trace["false_new_alerts_background"],
               "reason": trace["false_alarm_reason"]}
        for part, trace in pair.items()}
        for key, pair in result["traces"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
