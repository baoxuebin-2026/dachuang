#!/usr/bin/env python3
"""One-time J1 selection scorer; refuses to open source until lock is approved."""

from __future__ import annotations

import argparse
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

if __package__:
    from .build_stage9_b2_development_windows import audit_channel, file_sha256
    from .diagnose_stage9_b2_d1_development import trace_alarms
    from .prepare_stage9_b2_j1_candidate import score_from_saved_model
    from .run_stage9_b2_a_dev_comparison import ROOT, feature_matrix, score_alerts
else:
    from build_stage9_b2_development_windows import audit_channel, file_sha256
    from diagnose_stage9_b2_d1_development import trace_alarms
    from prepare_stage9_b2_j1_candidate import score_from_saved_model
    from run_stage9_b2_a_dev_comparison import ROOT, feature_matrix, score_alerts


CONFIG = ROOT / "configs/stage9_b2_j1_robust_candidate_v1.json"
LABEL_CONFIG = ROOT / "configs/stage9_b2_protocol_v1.json"
CANDIDATE = ROOT / "results/stage9_b2/j1_robust_candidate_development_v1.json"
LOCK = ROOT / "configs/stage9_b2_j1_selection_lock_v1.json"
OUTPUT = ROOT / "results/stage9_b2/j1_selection_once_v1.json"
COLS = ["year", "month", "day", "hour", "minute", "second", "MM256", "MM264"]


def load_selection(path: Path, source_sha: str, dates: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if file_sha256(path) != source_sha:
        raise ValueError("Source SHA mismatch")
    start, end = dates
    times, seconds, target, auxiliary = [], [], [], []
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["methane_data.csv"] or archive.testzip() is not None:
            raise ValueError("Unexpected archive/CRC")
        with archive.open("methane_data.csv") as stream:
            for chunk in pd.read_csv(stream, usecols=COLS, chunksize=150_000):
                day = pd.to_datetime(chunk[["year", "month", "day"]]).dt.strftime("%Y-%m-%d")
                if day.iloc[0] > end:
                    break
                chunk = chunk.loc[day.between(start, end)]
                if chunk.empty:
                    continue
                times.append(pd.to_datetime(chunk[COLS[:6]]).to_numpy(dtype="datetime64[s]").astype(np.int64))
                seconds.append(chunk.second.to_numpy(dtype=np.int8))
                target.append(chunk.MM256.to_numpy(dtype=np.float32))
                auxiliary.append(chunk.MM264.to_numpy(dtype=np.float32))
    if not times:
        raise ValueError("No selection-period rows")
    t = np.concatenate(times)
    if (str(np.datetime64(int(t[0]), "s"))[:10] != start
            or str(np.datetime64(int(t[-1]), "s"))[:10] != end):
        raise ValueError("Incomplete selection date coverage")
    return t, np.concatenate(seconds), np.concatenate(target), np.concatenate(auxiliary)


def robust_features(values: np.ndarray, rows: np.ndarray, median: float, hi: float) -> np.ndarray:
    valid = np.isfinite(values) & (values >= 0) & (values <= hi)
    now, lag = valid[rows], valid[rows - 300]
    both = now & lag
    return np.column_stack((np.where(now, values[rows], median),
                            np.where(both, values[rows] - values[rows - 300], 0.),
                            ~now, ~both)).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "data/raw/mendeley_yd7vw4c5mk/methane_data.zip")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise ValueError("One-time selection output must use the canonical locked path")
    if args.output.exists():
        raise FileExistsError("Selection already scored; never overwrite")
    # The lock is checked BEFORE opening the archive or constructing any labels.
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["selection_authorized"] is not True or not lock["human_decision_reference"]:
        raise PermissionError("Project confirmation of exact locked J1 candidate is pending")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    label = json.loads(LABEL_CONFIG.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if (file_sha256(CANDIDATE) != lock["candidate_sha256"]
            or file_sha256(CONFIG) != lock["protocol_sha256"]
            or file_sha256(LABEL_CONFIG) != candidate["label_protocol_sha256"]
            or candidate["protocol_sha256"] != file_sha256(CONFIG)
            or lock["selection_dates_inclusive"] != cfg["source_periods"]["one_time_selection_inclusive"]):
        raise ValueError("Candidate or protocol differs from approved lock")
    t, s, x, mm264 = load_selection(args.archive, cfg["source_sha256"],
                                     lock["selection_dates_inclusive"])
    summary, events, windows = audit_channel(t, s, x, label)
    rows = np.array([v["source_row"] for v in windows], dtype=np.int64)
    y = np.array([v["proxy_label"] for v in windows], dtype=np.int8)
    event_rows = np.array([v["source_row"] for v in events], dtype=np.int64)
    if not len(rows) or not len(event_rows):
        raise ValueError("Selection has no evaluable windows or proxy events")
    f0 = feature_matrix(x, rows)
    f1 = np.column_stack((f0, robust_features(mm264, rows,
                                                candidate["train_only_mm264_median"],
                                                cfg["mm264_valid_max_inclusive_pct_ch4"])))
    result = {
        "status": "one_time_selection_proxy_scoring; reserved_dates_not_scored",
        "lock_sha256": file_sha256(LOCK), "candidate_sha256": file_sha256(CANDIDATE),
        "protocol_sha256": file_sha256(CONFIG), "source_archive_sha256": file_sha256(args.archive),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "selection_source_rows": len(t), "selection_summary": summary,
        "evaluation": {},
    }
    for name, features in (("original_MM256_only_baseline", f0),
                           ("robust_MM256_plus_MM264", f1)):
        model = candidate["models"][name]
        scores = score_from_saved_model(features, model)
        threshold = model["training_threshold_budget_2"]
        traced = trace_alarms(t[rows], y, t[event_rows], scores, threshold, [60, 180])
        checked = score_alerts(rows, t[rows], y, event_rows, t[event_rows], scores, threshold)
        if (traced["matched_events_any_lead"] != checked["matched_events"]
                or traced["false_new_alerts_background"] != checked["false_new_alerts_background"]):
            raise ValueError("Original alert scorer disagrees with diagnostic trace")
        traced["false_per_24_background_hours"] = checked["false_per_24_background_hours"]
        result["evaluation"][name] = traced
    baseline = result["evaluation"]["original_MM256_only_baseline"]
    robust = result["evaluation"]["robust_MM256_plus_MM264"]
    baseline_long = {v["onset_local_naive"] for v in baseline["event_records"] if v["lead_seconds"] >= 60}
    robust_long = {v["onset_local_naive"] for v in robust["event_records"] if v["lead_seconds"] >= 60}
    extra = sorted(robust_long - baseline_long)
    enough = baseline["evaluable_events"] >= 10
    supports = (enough and len(robust_long) - len(baseline_long) >= 2
                and len(extra) >= 2 and len({x[:10] for x in extra}) >= 2
                and robust["false_per_24_background_hours"] <= 2
                and robust["false_per_24_background_hours"] <= baseline["false_per_24_background_hours"])
    result["limited_support_criterion"] = {
        "status": "limited_proxy_support" if supports else "not_supported" if enough else "insufficient_events",
        "evaluable_events": baseline["evaluable_events"],
        "additional_60s_event_onsets": extra,
        "lost_baseline_60s_event_onsets": sorted(baseline_long - robust_long),
        "net_60s_event_gain": len(robust_long) - len(baseline_long),
        "additional_60s_distinct_dates": len({x[:10] for x in extra}),
        "criterion_from_frozen_protocol": cfg["limited_support_criterion"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection_summary": summary,
                      "limited_support_criterion": result["limited_support_criterion"]}, indent=2))


if __name__ == "__main__":
    main()
