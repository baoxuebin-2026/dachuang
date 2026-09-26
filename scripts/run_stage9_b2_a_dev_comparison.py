#!/usr/bin/env python3
"""Compare three causal MM256 methods on a temporally blocked development split.

Training and internal validation are both within the already inspected
development dates. Selection/reserved dates are not used for fits or scores.
All features are functions of the primary sensor's past 300 seconds; the
published aligned series may itself contain forward-filled measurements.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
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
    from .build_stage9_b2_development_windows import load_development, file_sha256
else:
    from build_stage9_b2_development_windows import load_development, file_sha256


ROOT = Path(__file__).resolve().parents[1]
METHOD_CONFIG = ROOT / "configs/stage9_b2_a_dev_methods_v1.json"
LABEL_CONFIG = ROOT / "configs/stage9_b2_protocol_v1.json"
WINDOWS = ROOT / "data/processed/stage9_b2/development_MM256_windows.csv.gz"
EVENTS = ROOT / "data/processed/stage9_b2/development_MM256_events.csv.gz"
LABEL_REPORT = ROOT / "results/stage9_b2/development_windows_v1.json"
OUTPUT = ROOT / "results/stage9_b2/a_dev_comparison_v1.json"
SWEEP = ROOT / "results/stage9_b2/a_dev_tradeoff_v1.csv"


def read_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return pd.read_csv(stream)


def naive_seconds(ts: str) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000_000)


def feature_matrix(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0 or np.min(rows) < 300:
        raise ValueError("Every selected window needs at least 300 prior rows")
    cs = np.r_[0., np.cumsum(x, dtype=np.float64)]
    return np.column_stack((
        x[rows], x[rows] - x[rows - 60], x[rows] - x[rows - 300],
        (cs[rows + 1] - cs[rows - 59]) / 60,
        (cs[rows + 1] - cs[rows - 299]) / 300,
    ))


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    # Quantiles are always computed from training scores only.
    q = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))
    return np.r_[np.inf, q[::-1]]


def score_alerts(rows: np.ndarray, times: np.ndarray, labels: np.ndarray,
                 event_rows: np.ndarray, event_times: np.ndarray,
                 scores: np.ndarray, threshold: float, horizon: int = 360) -> dict:
    """Each new alarm matches at most one *future* event, never the same event twice."""
    if not (len(rows) == len(times) == len(labels) == len(scores)):
        raise ValueError("Window arrays differ in length")
    if len(rows) and (np.any(np.diff(rows) <= 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Window anchors must be strictly ordered")
    if len(event_rows) != len(event_times):
        raise ValueError("Event arrays differ in length")
    evaluable = (np.searchsorted(times, event_times, side="left")
                 > np.searchsorted(times, event_times - horizon, side="left"))
    active = scores >= threshold
    prev_active_adjacent = np.r_[False, active[:-1] & (np.diff(times) == 60)]
    new = active & ~prev_active_adjacent
    new_at = np.flatnonzero(new)
    matched = np.zeros(len(event_times), dtype=bool)
    lead = []
    background_false = 0
    duplicate_or_unmatched_positive = 0
    for k in new_at:
        j = np.searchsorted(event_times, times[k], side="right")
        while (j < len(event_times) and event_times[j] <= times[k] + horizon
               and (matched[j] or not evaluable[j])):
            j += 1
        if j < len(event_times) and event_times[j] <= times[k] + horizon and evaluable[j]:
            matched[j] = True
            lead.append(int(event_times[j] - times[k]))
        elif labels[k] == 0:
            background_false += 1
        else:
            duplicate_or_unmatched_positive += 1
    negative_minutes = int(np.sum(labels == 0))
    return {
        "evaluable_events": int(evaluable.sum()),
        "matched_events": int(matched.sum()),
        "new_alert_episodes": int(len(new_at)),
        "false_new_alerts_background": background_false,
        "duplicate_or_unmatched_positive_new_alerts": duplicate_or_unmatched_positive,
        "eligible_negative_minutes": negative_minutes,
        "false_per_24_background_hours": 1440 * background_false / negative_minutes if negative_minutes else None,
        "first_lead_seconds": lead,
    }


def report_without_lead_array(record: dict) -> dict:
    lead = record["first_lead_seconds"]
    return {**{k: v for k, v in record.items() if k != "first_lead_seconds"},
            "lead_seconds_median": float(np.median(lead)) if lead else None,
            "lead_seconds_min": min(lead) if lead else None,
            "lead_seconds_max": max(lead) if lead else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sweep", type=Path, default=SWEEP)
    args = parser.parse_args()
    if args.output.exists() or args.sweep.exists():
        raise FileExistsError("Existing method-comparison output")
    cfg = json.loads(METHOD_CONFIG.read_text(encoding="utf-8"))
    label_cfg = json.loads(LABEL_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(LABEL_REPORT.read_text(encoding="utf-8"))
    if (manifest["protocol_sha256"] != file_sha256(LABEL_CONFIG)
            or manifest["source_archive_sha256"] != label_cfg["source_sha256"]):
        raise ValueError("Label protocol/source manifest mismatch")
    by_file = {e["path"]: e for e in manifest["processed_files_ignored_by_git"]}
    for p in (WINDOWS, EVENTS):
        if file_sha256(p) != by_file[str(p.relative_to(ROOT))]["sha256"]:
            raise ValueError(f"Processed input differs from audited file: {p}")

    windows = read_csv_gz(WINDOWS)
    events = read_csv_gz(EVENTS)
    timestamps, _, targets = load_development(args.archive, label_cfg)
    x = targets[cfg["source_target"]]
    rows_all = windows.source_row.to_numpy(dtype=np.int64)
    labels_all = windows.proxy_label.to_numpy(dtype=np.int8)
    if not np.array_equal(
        pd.to_datetime(windows.prediction_local_naive).to_numpy(dtype="datetime64[s]").astype(np.int64),
        timestamps[rows_all],
    ):
        raise ValueError("Processed minute timestamp/source index mismatch")
    cut = naive_seconds(cfg["internal_validation_dates_inclusive"][0])
    train_mask = timestamps[rows_all] + cfg["split_guard_future_seconds"] < cut
    valid_mask = timestamps[rows_all] - cfg["split_guard_history_seconds"] >= cut
    if np.any(train_mask & valid_mask):
        raise ValueError("Temporal split overlaps")
    e_rows = events.source_row.to_numpy(dtype=np.int64)
    e_times = timestamps[e_rows]
    parts = {}
    for part, mask, event_mask in (
        ("train", train_mask, e_times < cut),
        ("internal_validation", valid_mask, e_times >= cut),
    ):
        rows = rows_all[mask]
        times = timestamps[rows]
        y = labels_all[mask]
        ev_rows = e_rows[event_mask]
        ev_times = e_times[event_mask]
        if not len(rows) or not len(ev_rows):
            raise ValueError(f"No valid windows or events in {part}")
        parts[part] = {
            "rows": rows, "times": times, "labels": y,
            "features": feature_matrix(x, rows),
            "event_rows": ev_rows, "event_times": ev_times,
        }
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=20260926))
    model.fit(parts["train"]["features"], parts["train"]["labels"])
    score_by_method = {}
    for part, data in parts.items():
        f = data["features"]
        score_by_method[part] = {
            "current_concentration": f[:, 0],
            "causal_slope_extrapolation": f[:, 0] + 360 * f[:, 2] / 300,
            "simple_logistic": model.predict_proba(f)[:, 1],
        }

    budgets = cfg["exploratory_false_new_alert_budgets_per_24_background_hours"]
    summary = {
        "status": "development_only_internal_validation; no selection/reserved scores",
        "method_config_sha256": file_sha256(METHOD_CONFIG),
        "label_protocol_sha256": file_sha256(LABEL_CONFIG),
        "source_archive_sha256": label_cfg["source_sha256"],
        "label_manifest_sha256": file_sha256(LABEL_REPORT),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "split": {}, "methods": {}, "model_coefficients_scaled": {},
    }
    for part, data in parts.items():
        summary["split"][part] = {
            "eligible_windows": int(len(data["rows"])),
            "positive_windows": int(data["labels"].sum()),
            "proxy_events": int(len(data["event_times"])),
            "evaluable_events": int(sum(np.any((data["times"] >= e - 360)
                                                  & (data["times"] < e)) for e in data["event_times"])),
            "first_window_local_naive": str(np.datetime64(int(data["times"][0]), "s")),
            "last_window_local_naive": str(np.datetime64(int(data["times"][-1]), "s")),
        }
    lr = model[-1]
    summary["model_coefficients_scaled"] = {
        "feature_names": cfg["features"],
        "coefficients": list(map(float, lr.coef_[0])),
        "intercept": float(lr.intercept_[0]),
        "classes": list(map(int, lr.classes_)),
    }
    sweep_rows = []
    for method, train_scores in score_by_method["train"].items():
        thresholds = candidate_thresholds(train_scores)
        train = parts["train"]
        valid = parts["internal_validation"]
        train_runs = []
        for threshold in thresholds:
            record = score_alerts(train["rows"], train["times"], train["labels"],
                                  train["event_rows"], train["event_times"], train_scores,
                                  float(threshold))
            train_runs.append(record)
            sweep_rows.append({
                "method": method, "threshold": float(threshold),
                "train_matched_events": record["matched_events"],
                "train_evaluable_events": record["evaluable_events"],
                "train_false_per_24h": record["false_per_24_background_hours"],
                "train_new_alerts": record["new_alert_episodes"],
            })
        summary["methods"][method] = {"threshold_candidates_from_train": len(thresholds),
                                      "budget_comparisons": {}}
        for budget in budgets:
            allowed = [i for i, record in enumerate(train_runs)
                       if record["false_per_24_background_hours"] is not None
                       and record["false_per_24_background_hours"] <= budget]
            best = min(allowed, key=lambda i: (
                -train_runs[i]["matched_events"],
                train_runs[i]["false_per_24_background_hours"],
                train_runs[i]["duplicate_or_unmatched_positive_new_alerts"],
                -thresholds[i],
            ))
            val_record = score_alerts(valid["rows"], valid["times"], valid["labels"],
                                      valid["event_rows"], valid["event_times"],
                                      score_by_method["internal_validation"][method], float(thresholds[best]))
            summary["methods"][method]["budget_comparisons"][str(budget)] = {
                "threshold_selected_from_train": (
                    float(thresholds[best]) if np.isfinite(thresholds[best]) else "always_silent"),
                "train": report_without_lead_array(train_runs[best]),
                "internal_validation": report_without_lead_array(val_record),
            }
    args.sweep.parent.mkdir(parents=True, exist_ok=True)
    with args.sweep.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    summary["development_train_sweep_csv"] = str(args.sweep.relative_to(ROOT))
    summary["development_train_sweep_sha256"] = file_sha256(args.sweep)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": summary["split"], "methods": summary["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
