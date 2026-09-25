#!/usr/bin/env python3
"""Posthoc Stage 7B diagnostics, with all Stage 7A detector parameters frozen."""

from __future__ import annotations

import csv
import json
import math
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_stage7_a import (csv_write, eligible_onsets, load_one_hz,
                                  score_sensors, sha256, timeslice_mask,
                                  transitions)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/stage7_b_posthoc_plan.json"
PLAN_SHA = "46934c2203452e7268f655bea8d8d8a366813e969018c736997784f14e43bd07"


def ethylene_only_onsets(runs, interval, prior_s=60, dwell_s=60, horizon_s=60):
    start, end = interval
    return [after["start_s"] for before, after in zip(runs, runs[1:])
            if before["state"] == "both_zero"
            and after["state"] == "target_zero_ethylene_present"
            and before["start_s"] >= start
            and before["duration_s"] >= prior_s
            and after["duration_s"] >= dwell_s
            and start <= after["start_s"]
            and after["start_s"] + horizon_s <= end]


def longest_above(values, threshold):
    longest = current = 0
    for value in values:
        current = current + 1 if np.isfinite(value) and value > threshold else 0
        longest = max(longest, current)
    return longest


def first_new_alarm(events, onset_s, horizon_s):
    return next((e for e in events if onset_s < e["evidence_start_s"]
                 and e["time_s"] <= onset_s + horizon_s), None)


def repeat_stat(onsets, period=200, tolerance=2):
    return [int(any(t != other and abs(abs(t - other) - period) <= tolerance
                    for other in onsets)) for t in onsets]


def main():
    if sha256(PLAN) != PLAN_SHA:
        raise ValueError("Stage 7B plan changed; freeze before running")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / plan["read_only_parent_run"]).read_text(encoding="utf-8"))
    config_path = ROOT / "configs/stage7_a_protocol_candidate.json"
    if sha256(config_path) != plan["parent_stage7a_frozen_config_sha256"]:
        raise ValueError("Stage 7A frozen config changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))["uci322"]
    source = ROOT / "data/raw/uci_322_original.zip"
    if sha256(source) != parent["source_uci322_sha256"]:
        raise ValueError("UCI archive checksum changed")
    audit_path = ROOT / "results/stage7_e1/uci322_source_audit.json"
    if sha256(audit_path) != parent["audit_uci322_sha256"]:
        raise ValueError("Source audit checksum changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with (ROOT / "results/stage7_a/uci322_summary.csv").open(newline="", encoding="utf-8") as f:
        original_summary = list(csv.DictReader(f))
    training = {x["file"]: x for x in parent["gas_training_parameters"]}
    blocks = {key: tuple(value) for key, value in config["intervals_s"].items()}
    summaries, events_out, detail_out, repeats_out, distribution_out = [], [], [], [], []
    with zipfile.ZipFile(source) as archive:
        for filename in plan["scope"]:
            runs = audit[filename]["runs"]
            series = load_one_hz(archive, filename, max(end for _, end in blocks.values()))
            frozen = training[filename]
            center = np.asarray(frozen["train_channel_center"])
            scale = np.asarray(frozen["train_channel_mad_scale"])
            channel = int(frozen["selected_channel_0_based"])
            for split, span in blocks.items():
                onset_groups = {
                    "target_gas": eligible_onsets(runs, span, 60, 60, 60),
                    "ethylene_only": ethylene_only_onsets(runs, span),
                }
                for kind, onsets in onset_groups.items():
                    for onset, matched in zip(onsets, repeat_stat(onsets)):
                        repeats_out.append({"file": filename, "split": split,
                                            "stimulus": kind, "setpoint_onset_s": onset,
                                            "has_same_type_neighbor_200_plusminus_2s": matched})
                for method in ("B0", "B1"):
                    row = next(x for x in original_summary if x["file"] == filename
                               and x["split"] == split and x["method"] == method
                               and float(x["quantile"]) == plan["main_quantile"])
                    threshold = float(row["threshold"])
                    score = score_sensors(series["features"], series["valid"], center,
                                          scale, method, channel)
                    alarms = transitions(score, threshold, span,
                                         config["evidence_consecutive_bins"])
                    negative = timeslice_mask(runs, span,
                                              config["negative_both_zero_min_prior_s"], len(score))
                    negative &= series["both_zero"] & series["valid"]
                    neg_scores = score[negative]
                    distribution_out.append({"file": filename, "split": split,
                                             "method": method,
                                             "negative_seconds": len(neg_scores),
                                             "negative_median": float(np.median(neg_scores)) if len(neg_scores) else "",
                                             "negative_p99": float(np.quantile(neg_scores, .99)) if len(neg_scores) else "",
                                             "negative_max": float(np.max(neg_scores)) if len(neg_scores) else "",
                                             "threshold": threshold})
                    if split == "train":
                        calibration = timeslice_mask(runs, span,
                                                     config["development_baseline_both_zero_min_prior_s"], len(score))
                        calibration &= series["both_zero"] & series["valid"]
                        calibration_scores = score[calibration]
                        distribution_out.append({"file": filename, "split": "train_calibration_60s",
                                                 "method": method,
                                                 "negative_seconds": len(calibration_scores),
                                                 "negative_median": float(np.median(calibration_scores)),
                                                 "negative_p99": float(np.quantile(calibration_scores, .99)),
                                                 "negative_max": float(np.max(calibration_scores)),
                                                 "threshold": threshold})
                    for kind, onsets in onset_groups.items():
                        found = 0
                        for onset in onsets:
                            first = first_new_alarm(alarms, onset, 60)
                            found += first is not None
                            left, right = math.ceil(onset), math.ceil(onset + 60)
                            window = score[left:right]
                            finite = window[np.isfinite(window)]
                            peak = float(max(finite)) if len(finite) else float("nan")
                            longest = longest_above(window, threshold)
                            events_out.append({"file": filename, "split": split,
                                               "method": method, "stimulus": kind,
                                               "onset_s": onset, "first_alarm_s": first["time_s"] if first else "",
                                               "delay_s": round(first["time_s"] - onset, 3) if first else "",
                                               "max_score_60s": peak,
                                               "longest_consecutive_gt_threshold_s": longest,
                                               "already_high_at_onset": int(np.isfinite(score[left - 1]) and
                                                                             score[left - 1] > threshold)
                                               if left > span[0] else 0})
                            if filename == "ethylene_CO.txt" and split == "test" and method == "B0" and kind == "target_gas":
                                detail_out.append({"setpoint_onset_s": onset,
                                                   "frozen_channel_0_based": channel,
                                                   "training_center_adc": float(center[channel]),
                                                   "training_mad_scale_adc": float(scale[channel]),
                                                   "frozen_threshold": threshold,
                                                   "max_score_60s": peak,
                                                   "threshold_minus_peak": threshold - peak,
                                                   "longest_consecutive_gt_threshold_s": longest,
                                                   "new_alarm_within_60s": int(first is not None)})
                        summaries.append({"file": filename, "split": split,
                                          "method": method, "stimulus": kind,
                                          "qualified_onsets": len(onsets),
                                          "new_alarms_60s": found,
                                          "threshold": threshold})
                        if kind == "target_gas" and found != int(row["detected_within_60s"]):
                            raise ValueError(f"Original target detection count mismatch: {filename}/{split}/{method}: {found} != {row['detected_within_60s']}")
    output = ROOT / "results/stage7_b"
    output.mkdir(exist_ok=True)
    for filename, rows in (("challenge_summary.csv", summaries),
                           ("challenge_events.csv", events_out),
                           ("co_b0_failure_details.csv", detail_out),
                           ("repeat_200s_diagnostic.csv", repeats_out),
                           ("negative_score_distributions.csv", distribution_out)):
        csv_write(output / filename, rows, list(rows[0]))
    (output / "run.json").write_text(json.dumps({
        "status": "posthoc_diagnostic_only", "stage7_b_plan_sha256": PLAN_SHA,
        "stage7_a_frozen_config_sha256": sha256(config_path),
        "stage7_a_parent_run_sha256": sha256(ROOT / plan["read_only_parent_run"]),
        "source_uci322_sha256": sha256(source),
        "script_sha256": sha256(Path(__file__)),
        "command": "python -m scripts.diagnose_stage7_b",
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "random_seed": None,
        "note": "Lab setpoint stimuli, not measured concentration or real coke-plant events; no threshold refit."
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in summaries:
        if row["split"] == "test":
            print(row)


if __name__ == "__main__":
    main()
