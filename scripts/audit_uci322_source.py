#!/usr/bin/env python3
"""Read-only source audit of UCI 322 setpoints; never treats them as measured gas."""

import argparse
import collections
import json
import zipfile
from pathlib import Path


def audit(stream):
    next(stream)  # header
    runs = []
    transitions = collections.Counter()
    rows = 0
    repeated_timestamps = 0
    first_time = last_time = None
    current = None
    run_start = None
    last_pair = None
    for raw in stream:
        parts = raw.split(maxsplit=3)
        if len(parts) != 4:
            raise ValueError(f"Unexpected row layout at row {rows + 1}")
        time, target, ethylene = map(float, parts[:3])
        if first_time is None:
            first_time = time
        if last_time is not None and time < last_time:
            raise ValueError(f"Decreasing time at row {rows + 1}")
        if last_time is not None and time == last_time:
            repeated_timestamps += 1
        last_time = time
        pair = (target, ethylene)
        state = "both_zero" if pair == (0.0, 0.0) else (
            "target_zero_ethylene_present" if target == 0.0 else "target_present"
        )
        if current is None:
            current, run_start = state, time
        elif current != state:
            runs.append({"state": current, "start_s": round(run_start, 2),
                         "end_s": round(time, 2), "duration_s": round(time - run_start, 2)})
            transitions[f"{current}->{state}"] += 1
            current, run_start = state, time
        if last_pair is not None and pair != last_pair:
            transitions["any_setpoint_change"] += 1
        last_pair = pair
        rows += 1
    if current is None:
        raise ValueError("Empty file")
    runs.append({"state": current, "start_s": round(run_start, 2),
                 "end_s": round(last_time, 2), "duration_s": round(last_time - run_start, 2)})
    by_state = {}
    for state in ("both_zero", "target_zero_ethylene_present", "target_present"):
        durations = [r["duration_s"] for r in runs if r["state"] == state]
        by_state[state] = {
            "runs": len(durations), "total_s": round(sum(durations), 2),
            "max_s": max(durations, default=0),
            "runs_at_least_120_s": sum(x >= 120 for x in durations),
        }
    return {"rows": rows, "first_s": first_time, "last_s": last_time,
            "repeated_timestamps": repeated_timestamps,
            "setpoint_transitions": dict(transitions), "state_runs": by_state,
            "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/uci_322_original.zip"))
    parser.add_argument("--out", type=Path, default=Path("results/stage7_e1/uci322_source_audit.json"))
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        report = {"dataset": "UCI 322", "state_definition":
                  "gas delivery setpoint; neither measured concentration nor accident label"}
        for name in ("ethylene_CO.txt", "ethylene_methane.txt"):
            with archive.open(name) as stream:
                report[name] = audit(stream)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("ethylene_CO.txt", "ethylene_methane.txt"):
        item = report[name]
        print(name, "rows", item["rows"], "duration_s", item["last_s"],
              "states", item["state_runs"], "transitions", item["setpoint_transitions"])


if __name__ == "__main__":
    main()
