#!/usr/bin/env python3
"""Compare S1 handling policies on the existing lab-gas/synthetic-person replay.

This is a software behavior comparison. There are no independent accident labels.
Run after the stage10 protocol has been committed, without changing stage6 files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.run_m1_uci309 import assign_splits, calibration, detector_series, load_trials, sha256
    from scripts.run_stage6_s1_a import fuse, gas_states, simulate
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_m1_uci309 import assign_splits, calibration, detector_series, load_trials, sha256
    from run_stage6_s1_a import fuse, gas_states, simulate


STRATEGIES = (
    "S1_full", "gas_only", "simple_OR", "simple_AND", "no_zone_gate",
    "no_age_gate", "hold_missing_gas_10s", "hold_missing_person_10s",
)


def number(value: str) -> int | None:
    return int(value[1:]) if value != "unknown" else None


def level(gas: int | None, person: int | None, *, same_zone: bool = True) -> int | None:
    return fuse(gas, person, same_zone=same_zone)["joint_level"] if gas is not None and person is not None and same_zone else max(
        (value for value in (gas, person) if value is not None), default=None
    )


def evaluate(timeline: list[dict], states: dict[int, tuple[int | None, str]],
             fault_times: set[int]) -> dict[str, dict[str, int]]:
    counts = {name: defaultdict(int) for name in STRATEGIES}
    previous_gas: tuple[int, int] | None = None
    previous_person: tuple[int, int] | None = None
    for row in timeline:
        t = row["completed_time_s"]
        gas = number(row["gas_evidence"])
        person = number(row["person_relation"])
        if gas is not None:
            previous_gas = (gas, t)
        if person is not None:
            previous_person = (person, t)
        held_gas = (previous_gas[0] if previous_gas and t - previous_gas[1] <= 10 else None)
        held_person = (previous_person[0] if previous_person and t - previous_person[1] <= 10 else None)
        raw_gas = states[t][0]
        age_gas = raw_gas if row["Qg"] == "simulated_stale" else gas
        missing_gas = held_gas if row["Qg"] == "simulated_dropout" else gas
        missing_person = held_person if row["Qp"] == "simulated_dropout" else person
        zone = row["same_zone"]
        original_eligible = gas is not None and person is not None and zone
        known = [v for v in (gas, person) if v is not None]
        candidates = {
            "S1_full": row["joint_level"] if original_eligible else row["known_lower_bound"],
            "gas_only": gas,
            "simple_OR": max(known, default=None),
            "simple_AND": 3 if gas == 2 and person in (1, 2) else max(known, default=None),
            "no_zone_gate": level(gas, person),
            "no_age_gate": level(age_gas, person, same_zone=zone),
            "hold_missing_gas_10s": level(missing_gas, person, same_zone=zone),
            "hold_missing_person_10s": level(gas, missing_person, same_zone=zone),
        }
        for name, output in candidates.items():
            item = counts[name]
            item["steps"] += 1
            item["ineligible_steps"] += int(not original_eligible)
            item["l3_seconds"] += int(output == 3)
            item["l3_on_ineligible_seconds"] += int(output == 3 and not original_eligible)
            item["l3_in_fault_window_seconds"] += int(output == 3 and t in fault_times)
            item["unknown_output_seconds"] += int(output is None)
        counts["S1_full"]["unknown_joint_seconds"] += int(row["unknown"])
        if ((gas == 2 or person == 2) and row["known_lower_bound"] != 2
                and row["joint_level"] != 3):
            counts["S1_full"]["lost_known_l2_seconds"] += 1
    for values in counts.values():
        values["unknown_joint_seconds"] += 0
        values["lost_known_l2_seconds"] += 0
    return {name: dict(values) for name, values in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage6_s1_fixed_a.json"))
    parser.add_argument("--out", type=Path, default=Path("results/stage10_s1_comparison"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256(args.archive) != cfg["data_sha256"]:
        raise ValueError("UCI 309 archive differs from the frozen stage6 configuration")
    original = json.loads(Path("results/m1_uci309/run.json").read_text(encoding="utf-8"))
    if original["run_id"] != cfg["m1_run_id"] or sha256(Path("scripts/run_m1_uci309.py")) != original["script_sha256"]:
        raise ValueError("M1 source or run differs from the frozen result")
    selected = [entry for entry in original["parameters"] if all(
        entry[key] == value for key, value in cfg["m1_setting"].items() if key != "split")]
    if len(selected) != 1:
        raise ValueError("Expected one frozen M1 selection")
    selected = selected[0]
    trials = load_trials(args.archive)
    manifest = pd.DataFrame(assign_splits(trials)).sort_values("trial").reset_index(drop=True)
    previous_manifest = pd.read_csv("results/m1_uci309/split_manifest.csv").sort_values("trial").reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest, previous_manifest)
    cal = calibration(trials, "gas", "A")
    if not (np.array_equal(cal["center"], selected["global_center"])
            and np.array_equal(cal["scale"], selected["scale"])):
        raise ValueError("M1 calibration differs from its registered values")
    test = sorted((trial for trial in trials if trial.split == "test"), key=lambda trial: trial.name)
    if len(test) != 60:
        raise ValueError("Expected all 60 existing gas test files")
    fault_times = set(range(*cfg["overlay_time_half_open_s"]))
    rows = []
    for trial in test:
        score, valid = detector_series(trial, cal, "multi")
        for scenario, setup in cfg["scenarios"].items():
            original_counts, timeline = simulate(
                score, valid, selected, scenario, cfg, buffer_m=cfg["baseline_buffer_m"],
                error_bound_m=0.0, duration_s=cfg["nominal_duration_s"], include_timeline=True,
            )
            missing = fault_times if setup["overlay"] == "gas_missing_100_109" else set()
            states = gas_states(score, valid, selected["threshold"], cfg["nominal_duration_s"], missing)
            for name, values in evaluate(timeline, states, fault_times).items():
                if name == "S1_full" and (
                    values["l3_seconds"] != original_counts["joint_l3_seconds"]
                    or values["unknown_joint_seconds"] != original_counts["unknown_seconds"]
                    or values.get("lost_known_l2_seconds", 0) != 0
                ):
                    raise AssertionError(f"S1 replay mismatch: {trial.name}, {scenario}")
                rows.append({"trial": trial.name, "scenario": scenario, "strategy": name, **values})
    table = pd.DataFrame(rows).fillna(0)
    integer_columns = [column for column in table if column not in ("trial", "scenario", "strategy")]
    table[integer_columns] = table[integer_columns].astype(int)
    summary = table.groupby(["scenario", "strategy"], sort=True).agg(
        gas_files=("trial", "nunique"),
        steps=("steps", "sum"),
        ineligible_steps=("ineligible_steps", "sum"),
        files_with_l3=("l3_seconds", lambda values: int((values > 0).sum())),
        l3_seconds=("l3_seconds", "sum"),
        l3_on_ineligible_seconds=("l3_on_ineligible_seconds", "sum"),
        l3_in_fault_window_seconds=("l3_in_fault_window_seconds", "sum"),
        unknown_joint_seconds=("unknown_joint_seconds", "sum"),
        lost_known_l2_seconds=("lost_known_l2_seconds", "sum"),
    ).reset_index()
    previous = pd.read_csv("results/stage6_s1_a/scenario_aggregate.csv").set_index("scenario")
    current = summary[summary.strategy == "S1_full"].set_index("scenario")
    for scenario in cfg["scenarios"]:
        for current_column, prior_column in (
            ("gas_files", "trials"), ("files_with_l3", "trials_with_joint_L3"),
            ("l3_seconds", "total_L3_seconds_across_replays"),
            ("unknown_joint_seconds", "unknown_seconds_across_replays"),
        ):
            if current.loc[scenario, current_column] != previous.loc[scenario, prior_column]:
                raise AssertionError(f"Registered stage6 aggregate mismatch: {scenario}, {current_column}")
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "per_file.csv", index=False, lineterminator="\n")
    summary.to_csv(args.out / "summary.csv", index=False, lineterminator="\n")
    metadata = {
        "run_id": "stage10-s1-comparison-20260927-v1",
        "status": "completed_software_behavior_comparison_not_independent_safety_validation",
        "protocol_sha256": sha256(Path("docs/stage10-s1-comparison-protocol.md")),
        "stage6_config_sha256": sha256(args.config),
        "uci309_archive_sha256": sha256(args.archive),
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "gas_files": len(test), "scenario_count": len(cfg["scenarios"]),
        "strategies": list(STRATEGIES),
        "registered_stage6_aggregates_match": True,
        "limitation": "One reused artificial path per scenario and already viewed gas test files; no independent risk, accident or same-site camera truth",
    }
    (args.out / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
