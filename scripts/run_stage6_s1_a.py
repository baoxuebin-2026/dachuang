#!/usr/bin/env python3
"""Stage 6 exploratory S1 replay: UCI 309 laboratory gas + simulated people.

python scripts/run_stage6_s1_a.py --archive data/raw/uci_309_original.zip \
    --config configs/stage6_s1_fixed_a.json --out results/stage6_s1_a

No laboratory gas data are presented as same-site measurements of a person.
The L0-L3 matrix is a simulated handling policy, not an accident classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.run_m1_uci309 import (assign_splits, calibration, detector_series,
                                       evidence_events, load_trials, score_events, sha256)
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_m1_uci309 import (assign_splits, calibration, detector_series, evidence_events,
                               load_trials, score_events, sha256)


MATRIX = ((0, 1, 2), (1, 2, 2), (2, 3, 3))
SOURCE_GAS = "UCI309_laboratory"
SOURCE_PERSON = "simulation"


def signed_distance(x: float, y: float, zone: dict) -> float:
    """Euclidean signed ground distance to the *rectangle* boundary, inside <=0."""
    left, right, bottom, top = (zone[k] for k in ("x_min", "x_max", "y_min", "y_max"))
    dx = max(left - x, 0.0, x - right)
    dy = max(bottom - y, 0.0, y - top)
    if dx > 0 or dy > 0:
        return math.hypot(dx, dy)
    return -min(x - left, right - x, y - bottom, top - y)


def locate(distance: float, buffer_m: float, error_bound_m: float) -> tuple[int | None, str]:
    """Classify only when every point within the simulated error bound agrees."""
    low, high = distance - error_bound_m, distance + error_bound_m
    if high <= 0:
        return 2, "valid"
    if low > 0 and high <= buffer_m:
        return 1, "valid"
    if low > buffer_m:
        return 0, "valid"
    return None, "boundary_uncertain"


def position_at(frames: list[list[float]], seconds: int) -> tuple[float, float]:
    if seconds < frames[0][0] or seconds > frames[-1][0]:
        raise ValueError("Outside declared person trajectory")
    for (t0, x0, y0), (t1, x1, y1) in zip(frames, frames[1:]):
        if t0 <= seconds <= t1:
            fraction = (seconds - t0) / (t1 - t0)
            return x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)
    raise ValueError("Invalid trajectory keyframes")


def unit_disk_offsets(trajectory_name: str, times: range, seed: int) -> dict[int, tuple[float, float]]:
    """Common noise draws shared across all gas trials and sensitivity settings."""
    digest = hashlib.sha256(f"{seed}:{trajectory_name}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    angle = rng.uniform(0, 2 * np.pi, len(times))
    radius = np.sqrt(rng.uniform(0, 1, len(times)))
    return {t: (float(radius[i] * np.cos(angle[i])), float(radius[i] * np.sin(angle[i])))
            for i, t in enumerate(times)}


def gas_states(score: np.ndarray, valid: np.ndarray, threshold: float, duration_s: int,
               missing_at: set[int] | None = None) -> dict[int, tuple[int | None, str]]:
    """Output at the *completed* second, reset evidence on missing data."""
    result = {}
    consecutive = 0
    missing_at = missing_at or set()
    for n in range(25, 240):
        t = n + 1
        if not valid[n] or not np.isfinite(score[n]) or t in missing_at:
            result[t] = (None, "simulated_dropout" if t in missing_at else "laboratory_invalid")
            consecutive = 0
            continue
        consecutive = consecutive + 1 if score[n] > threshold else 0
        result[t] = (2 if consecutive >= duration_s else 1 if consecutive else 0, "valid")
    return result


def fuse(gas: int | None, person: int | None, *, same_zone: bool,
         gas_quality: str = "valid", person_quality: str = "valid") -> dict:
    """Keep independent known evidence even when a joint conclusion is impossible."""
    valid_gas = gas_quality == "valid" and gas is not None
    valid_person = person_quality == "valid" and person is not None
    gas_level = (0, 1, 2)[gas] if valid_gas else None
    person_level = (0, 1, 2)[person] if valid_person else None
    known_levels = [v for v in (gas_level, person_level) if v is not None]
    if valid_gas and valid_person and same_zone:
        return {"joint_level": MATRIX[gas][person], "gas_level": gas_level,
                "person_level": person_level, "known_lower_bound": max(known_levels),
                "unknown": False, "paired_under_assumption": True,
                "uncertainty_reason": ""}
    causes = []
    if not valid_gas:
        causes.append(f"gas_{gas_quality}")
    if not valid_person:
        causes.append(f"person_{person_quality}")
    if not same_zone:
        causes.append("zone_mismatch")
    return {"joint_level": None, "gas_level": gas_level, "person_level": person_level,
            "known_lower_bound": max(known_levels) if known_levels else None,
            "unknown": True, "paired_under_assumption": False,
            "uncertainty_reason": "+".join(causes)}


def simulate(score: np.ndarray, valid: np.ndarray, selected: dict, scenario: str,
             cfg: dict, *, buffer_m: float, error_bound_m: float, duration_s: int,
             include_timeline: bool = False) -> tuple[dict, list[dict]]:
    setup = cfg["scenarios"][scenario]
    frames = cfg["keyframes_xy"][setup["trajectory"]]
    times = range(setup["inclusive_time"][0], setup["inclusive_time"][1] + 1)
    fault_times = set(range(*cfg["overlay_time_half_open_s"]))
    overlay = setup["overlay"]
    missing = fault_times if overlay == "gas_missing_100_109" else set()
    states = gas_states(score, valid, float(selected["threshold"]), duration_s, missing)
    offsets = unit_disk_offsets(setup["trajectory"], times, cfg["position_noise_seed"])
    target_zone = cfg["control_zone"]
    entries: list[dict] = []
    counts = {"steps": 0, "g2_seconds": 0, "p2_seconds": 0,
              "joint_l3_seconds": 0, "first_joint_l3_s": None,
              "pre_release_joint_l3_seconds": 0, "unknown_seconds": 0,
              "paired_seconds": 0, "zone_mismatch_seconds": 0,
              "boundary_uncertain_seconds": 0, "gas_invalid_seconds": 0,
              "person_invalid_seconds": 0, "person_p2_gas_missing_lower_bound_fail": 0,
              "gas_g2_person_missing_lower_bound_fail": 0}
    for t in times:
        x, y = position_at(frames, t)
        ux, uy = offsets[t]
        observed_x, observed_y = x + ux * error_bound_m, y + uy * error_bound_m
        signed_m = signed_distance(observed_x, observed_y, target_zone)
        p, pq = locate(signed_m, buffer_m, error_bound_m)
        g, gq = states[t]
        gas_age_s = 2 if overlay == "gas_age_2s_100_109" and t in fault_times else 0
        if gas_age_s > cfg["time_age_tolerance_s"]:
            gq = "simulated_stale"
        if overlay == "person_missing_100_109" and t in fault_times:
            p, pq = None, "simulated_dropout"
        zone_id = "zone_B" if overlay == "person_zone_B_100_109" and t in fault_times else target_zone["zone_id"]
        same_zone = zone_id == cfg["coverage_zone_id_assumption"]
        out = fuse(g, p, same_zone=same_zone, gas_quality=gq, person_quality=pq)
        counts["steps"] += 1
        counts["g2_seconds"] += int(g == 2 and gq == "valid")
        counts["p2_seconds"] += int(p == 2 and pq == "valid")
        counts["paired_seconds"] += int(out["paired_under_assumption"])
        counts["unknown_seconds"] += int(out["unknown"])
        counts["boundary_uncertain_seconds"] += int(pq == "boundary_uncertain")
        counts["gas_invalid_seconds"] += int(gq != "valid")
        counts["person_invalid_seconds"] += int(pq != "valid")
        counts["zone_mismatch_seconds"] += int(not same_zone)
        if out["joint_level"] == 3:
            counts["joint_l3_seconds"] += 1
            if counts["first_joint_l3_s"] is None:
                counts["first_joint_l3_s"] = t
            if t <= cfg["laboratory_program_for_offline_scoring_only_s"]["release_start"]:
                counts["pre_release_joint_l3_seconds"] += 1
        if p == 2 and pq == "valid" and gq != "valid":
            counts["person_p2_gas_missing_lower_bound_fail"] += int(out["known_lower_bound"] != 2)
        if g == 2 and gq == "valid" and pq != "valid":
            counts["gas_g2_person_missing_lower_bound_fail"] += int(out["known_lower_bound"] != 2)
        if include_timeline:
            entries.append({"scenario": scenario, "completed_time_s": t,
                            "gas_source": SOURCE_GAS, "person_source": SOURCE_PERSON,
                            "sim_true_x_m": x, "sim_true_y_m": y,
                            "sim_observed_x_m": None if pq == "simulated_dropout" else observed_x,
                            "sim_observed_y_m": None if pq == "simulated_dropout" else observed_y,
                            "sim_observed_signed_distance_m": None if pq == "simulated_dropout" else signed_m,
                            "position_error_bound_m": error_bound_m,
                            "gas_score_from_lab": float(score[t-1]),
                            "gas_evidence": f"G{g}" if gq == "valid" else "unknown",
                            "person_relation": f"P{p}" if pq == "valid" else "unknown",
                            "Qg": gq, "Qp": pq, "person_zone_id": zone_id,
                            "coverage_zone_id_assumption": cfg["coverage_zone_id_assumption"],
                            "same_zone": same_zone, "gas_age_s": gas_age_s, **out})
    if counts["person_p2_gas_missing_lower_bound_fail"] or counts["gas_g2_person_missing_lower_bound_fail"]:
        raise AssertionError("A missing input suppressed independent known priority")
    if counts["paired_seconds"] + counts["unknown_seconds"] != counts["steps"]:
        raise AssertionError("Output must be complete/known or explicitly unknown")
    return counts, entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/stage6_s1_a"))
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256(args.archive) != cfg["data_sha256"]:
        raise ValueError("UCI309 original archive hash differs from the frozen configuration")
    original = json.loads(Path("results/m1_uci309/run.json").read_text(encoding="utf-8"))
    if original["run_id"] != cfg["m1_run_id"] or sha256(Path("scripts/run_m1_uci309.py")) != original["script_sha256"]:
        raise ValueError("M1 provenance differs from the frozen run")
    selected = [entry for entry in original["parameters"] if all(
        entry[key] == value for key, value in cfg["m1_setting"].items() if key != "split")]
    if len(selected) != 1:
        raise ValueError("Expected exactly one original A / multi / gas selection")
    selected = selected[0]
    all_trials = load_trials(args.archive)
    manifest = pd.DataFrame(assign_splits(all_trials)).sort_values("trial").reset_index(drop=True)
    prior_manifest = pd.read_csv("results/m1_uci309/split_manifest.csv").sort_values("trial").reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest, prior_manifest)
    cal = calibration(all_trials, "gas", "A")
    if not (np.allclose(cal["center"], selected["global_center"], atol=0, rtol=0)
            and np.allclose(cal["scale"], selected["scale"], atol=0, rtol=0)):
        raise ValueError("M1 calibration differs from the registered selection")
    baseline = pd.read_csv("results/m1_uci309/file_results.csv")
    baseline = baseline.loc[(baseline["mode"] == "A") & (baseline.source == "gas")
                            & (baseline.method == "multi") & (baseline.split == "test")
                            & (baseline.control == "real_laboratory_data")].set_index("trial")
    test = sorted((t for t in all_trials if t.split == "test"), key=lambda t: t.name)
    if len(test) != 60 or len(baseline) != 60:
        raise ValueError("All original test trials must be replayed, without filtering by outcome")
    example_trials = {test[0].name, "011_Et_n_Me_L"}
    assert example_trials <= set(t.name for t in test)
    rows, sensitivity, examples = [], [], []
    for trial in test:
        score, valid = detector_series(trial, cal, "multi")
        previous = score_events(evidence_events(score, valid, selected["threshold"], 3))
        for col in ("false_file", "detected", "detected_within_60_s", "first_detection_s"):
            actual = previous[col]
            expected = baseline.loc[trial.name, col]
            if (pd.isna(expected) and actual is not None) or (not pd.isna(expected) and actual != expected):
                raise ValueError(f"Gas replay differs from registered M1 result: {trial.name} {col}")
        for scenario in cfg["scenarios"]:
            result, timeline = simulate(score, valid, selected, scenario, cfg,
                                        buffer_m=cfg["baseline_buffer_m"], error_bound_m=0.0,
                                        duration_s=cfg["nominal_duration_s"],
                                        include_timeline=trial.name in example_trials)
            rows.append({"trial": trial.name, "configuration": trial.config,
                         "scenario": scenario, "gas_source": SOURCE_GAS,
                         "person_source": SOURCE_PERSON, "gas_detected_during_release": previous["detected"],
                         "gas_false_file_before_release": previous["false_file"], **result})
            examples.extend({"trial": trial.name, **entry} for entry in timeline)
        for duration in cfg["gas_consecutive_sensitivity_s"]:
            for buffer_m in cfg["buffer_sensitivity_m"]:
                for bound in cfg["position_error_bound_sensitivity_m"]:
                    result, _ = simulate(score, valid, selected, "S2_possible_overlap", cfg,
                                         buffer_m=buffer_m, error_bound_m=bound,
                                         duration_s=duration)
                    sensitivity.append({"trial": trial.name, "scenario": "S2_possible_overlap",
                                        "duration_s": duration, "buffer_m": buffer_m,
                                        "position_error_bound_m": bound, **result})
    args.out.mkdir(parents=True, exist_ok=True)
    trial_table = pd.DataFrame(rows)
    trial_table.to_csv(args.out / "trial_scenario_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(examples).to_csv(args.out / "illustrative_timeline.csv", index=False, lineterminator="\n")
    sensitivity_table = pd.DataFrame(sensitivity)
    sensitivity_table.to_csv(args.out / "sensitivity_by_trial.csv", index=False, lineterminator="\n")
    grouped = sensitivity_table.groupby(["duration_s", "buffer_m", "position_error_bound_m"], sort=True)
    aggregate = grouped.agg(trials=("trial", "nunique"), trials_with_joint_L3=("joint_l3_seconds", lambda x: int((x > 0).sum())),
                            total_L3_seconds_across_replays=("joint_l3_seconds", "sum"),
                            unknown_seconds_across_replays=("unknown_seconds", "sum"),
                            boundary_uncertain_seconds_across_replays=("boundary_uncertain_seconds", "sum"))
    aggregate.reset_index().to_csv(args.out / "sensitivity_aggregate.csv", index=False, lineterminator="\n")
    canonical = trial_table.groupby("scenario", sort=True).agg(
        trials=("trial", "nunique"), trials_with_joint_L3=("joint_l3_seconds", lambda x: int((x > 0).sum())),
        total_L3_seconds_across_replays=("joint_l3_seconds", "sum"),
        unknown_seconds_across_replays=("unknown_seconds", "sum"),
        paired_seconds_across_replays=("paired_seconds", "sum"),
        pre_release_joint_L3_seconds=("pre_release_joint_l3_seconds", "sum"))
    canonical.reset_index().to_csv(args.out / "scenario_aggregate.csv", index=False, lineterminator="\n")
    summary = {"run_status": "exploratory_lab_gas_plus_simulated_person_replay",
               "original_uci309_sha256": sha256(args.archive),
               "frozen_config_sha256": sha256(args.config),
               "frozen_m1_run_id": original["run_id"],
               "gas_threshold": selected["threshold"],
               "script_sha256": sha256(Path(__file__)),
               "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
               "trial_count": len(test), "scenario_count": len(cfg["scenarios"]),
               "synthetic_scenarios_per_gas_file": list(cfg["scenarios"]),
               "example_trial_ids": sorted(example_trials),
               "gas_replay_matches_all_60_registered_pilot_file_scores": True,
               "nominal_scenario_totals": canonical.reset_index().to_dict(orient="records"),
               "sensitivity_settings": len(aggregate),
               "unit_note": "60 laboratory gas files replayed with fixed synthetic paths; no independent real people, accidents or camera measurements",
               "prior_holdout_note": "M1 test files were already viewed and rescored; this is not a fresh blind test",
               "recovery_note": "0-239s source bins only; original post-release recovery 240-300s was not replayed"}
    (args.out / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(canonical.to_string())
    print("Completed 60 gas files, 8 fixed scenario types and 27 sensitivity settings")


if __name__ == "__main__":
    main()
