#!/usr/bin/env python3
"""Frozen B1/P1 causal rule comparison, split into selection and locked phases.

The first phase refuses to read May 14-15 methane values. The locked phase
checks the saved selection and program digests before it reads that period.
Original source CSVs must come from wsdaniels/DLQ at the pinned commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage9_b1"
SOURCE_SHA = {
    "ADED_data_clean_SAMPLE.csv": "fb1fd21e77fef2ad508e2de2697743c1bb010b616457cde29a1480bf79d7c26b",
    "leak_data.csv": "20792a9b8c318ff3490c47f8e071b2dd22c69caeb5ce3356a4a6f6298e048b6a",
}
NAMES = ["E", "N", "NE", "NW", "S", "SE", "SW", "W"]
DEV = ("2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12")
SEL = ("2022-05-13",)
TEST = ("2022-05-14", "2022-05-15")
QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_new_json(path: Path, obj: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_paths(source_dir: Path) -> tuple[Path, Path]:
    for filename, digest in SOURCE_SHA.items():
        if sha256(source_dir / filename) != digest:
            raise ValueError(f"Unrecognized raw source: {filename}")
    return source_dir / "ADED_data_clean_SAMPLE.csv", source_dir / "leak_data.csv"


def prepare(source_dir: Path, days: tuple[str, ...]) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    observations, release_csv = source_paths(source_dir)
    # Selection phase parses methane values only on development + selection
    # dates. The file digest reads all bytes; test values are never parsed into
    # numbers or used in selection. Source remains an external raw file.
    day_set = set(days)
    selected_rows = []
    with observations.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        time_col, name_col, methane_col = (header.index(key) for key in ("time", "name", "methane"))
        for row in reader:
            if row[time_col][:10] in day_set:
                selected_rows.append((row[time_col], row[name_col], float(row[methane_col])))
    frame = pd.DataFrame(selected_rows, columns=["time", "name", "methane"])
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    start = pd.Timestamp(min(days), tz="UTC")
    end = pd.Timestamp(max(days), tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    grid = pd.date_range(start, end, freq="min", tz="UTC")
    wide = frame.pivot_table(index="time", columns="name", values="methane", aggfunc="median").reindex(grid).reindex(columns=NAMES)
    raw = wide.to_numpy(dtype=float)
    coverage = np.isfinite(raw).sum(axis=1) >= 6
    day_strings = np.asarray(grid.strftime("%Y-%m-%d"))
    warm = np.asarray(grid >= grid.floor("D") + pd.Timedelta(minutes=60))
    residual = np.full_like(raw, np.nan)
    recent = np.full_like(raw, np.nan)
    for day in days:
        idx = np.flatnonzero(day_strings == day)
        chunk = wide.iloc[idx]
        baseline = chunk.shift(1).rolling(60, min_periods=45).median()
        r = (chunk - baseline).clip(lower=0)
        residual[idx] = r.to_numpy()
        z = r.rolling(5, min_periods=1).max().where(r.notna())
        recent[idx] = z.to_numpy()
    single = {f"single_{name}": residual[:, i].copy() for i, name in enumerate(NAMES)}
    max_score = np.full(len(grid), np.nan)
    second_score = np.full(len(grid), np.nan)
    for row in range(len(grid)):
        available = residual[row, np.isfinite(residual[row])]
        if available.size >= 6:
            max_score[row] = float(np.max(available))
        v = recent[row, np.isfinite(recent[row])]
        if v.size >= 6:
            second_score[row] = float(np.partition(v, -2)[-2])
    scores = single | {"maximum": max_score, "two_point_5min": second_score}
    common = coverage & warm
    for key in scores:
        scores[key][~common] = np.nan

    log = pd.read_csv(release_csv, parse_dates=["tc_ExpStartDatetime", "tc_ExpEndDatetime"])
    left, right = grid[0], grid[-1]
    rel = log.loc[(log.tc_ExpStartDatetime <= right) & (log.tc_ExpEndDatetime >= left)].sort_values("tc_ExpStartDatetime")
    if (rel.tc_ExpEPCount != 1).any():
        raise ValueError("Unexpected multisource release in the selected period")
    active = np.zeros(len(grid), dtype=bool)
    washout = np.zeros(len(grid), dtype=bool)
    boundary = np.zeros(len(grid), dtype=bool)
    for e in rel.itertuples():
        active |= (grid >= e.tc_ExpStartDatetime) & (grid <= e.tc_ExpEndDatetime)
        washout |= (grid > e.tc_ExpEndDatetime) & (grid <= e.tc_ExpEndDatetime + pd.Timedelta(minutes=30))
        boundary |= (grid >= e.tc_ExpStartDatetime - pd.Timedelta(minutes=1)) & (grid < e.tc_ExpStartDatetime)
    eligible = ~active & ~washout & ~boundary & common
    return grid, scores, eligible, rel, active, washout


def alarms(score: np.ndarray, threshold: float, grid: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    state = np.full(score.size, -1, dtype=np.int8)
    starts = np.zeros(score.size, dtype=bool)
    prior = 0
    alarming = False
    last_day = None
    for j, value in enumerate(score):
        day = grid[j].date()
        if day != last_day:
            prior, alarming, last_day = 0, False, day
        if not np.isfinite(value):
            prior, alarming = 0, False
            continue
        evidence = value >= threshold
        prior = prior + 1 if evidence else 0
        new_alarm = evidence and prior >= 3
        state[j] = 1 if new_alarm else 0
        starts[j] = new_alarm and not alarming
        alarming = new_alarm
    return state, starts


def evaluate(score: np.ndarray, threshold: float, grid: pd.DatetimeIndex, eligible: np.ndarray,
             releases: pd.DataFrame, active: np.ndarray, washout: np.ndarray,
             event_start_days: tuple[str, ...]) -> dict:
    state, starts = alarms(score, threshold, grid)
    days = np.isin(np.asarray(grid.strftime("%Y-%m-%d")), event_start_days)
    background = eligible & days & (state != -1)
    background_n = int(background.sum())
    fp = int((starts & background).sum())
    occupied = int(((state == 1) & background).sum())
    event_rows = []
    duplicates = 0
    for e in releases.itertuples():
        if e.tc_ExpStartDatetime.strftime("%Y-%m-%d") not in event_start_days:
            continue
        candidates = np.flatnonzero(starts & (grid >= e.tc_ExpStartDatetime) & (grid <= e.tc_ExpEndDatetime))
        duplicates += max(0, len(candidates) - 1)
        first = grid[candidates[0]] if len(candidates) else None
        delay = (first - e.tc_ExpStartDatetime).total_seconds() / 60 if first is not None else None
        event_rows.append({
            "release_id": str(e.tc_EPID),
            "start_utc": e.tc_ExpStartDatetime.isoformat(),
            "end_utc": e.tc_ExpEndDatetime.isoformat(),
            "first_new_alert_utc": first.isoformat() if first is not None else None,
            "delay_minutes": round(delay, 3) if delay is not None else None,
            "within_30m": bool(delay is not None and delay <= 30),
        })
    delays = [r["delay_minutes"] for r in event_rows if r["delay_minutes"] is not None]
    starts_in_days = starts & days
    prior_30 = np.zeros(len(grid), dtype=bool)
    for e in releases.itertuples():
        if e.tc_ExpStartDatetime.strftime("%Y-%m-%d") in event_start_days:
            prior_30 |= (grid >= e.tc_ExpStartDatetime - pd.Timedelta(minutes=30)) & (grid < e.tc_ExpStartDatetime)
    return {
        "release_events": len(event_rows),
        "timely_30m": sum(e["within_30m"] for e in event_rows),
        "any_during_release": len(delays),
        "median_detected_delay_minutes": float(np.median(delays)) if delays else None,
        "event_details": event_rows,
        "repeat_alert_starts_during_same_release": duplicates,
        "eligible_background_minutes": background_n,
        "false_alert_starts": fp,
        "false_alerts_per_24h": round(fp * 1440 / background_n, 5) if background_n else None,
        "background_alarm_occupied_minutes": occupied,
        "background_alarm_occupancy": round(occupied / background_n, 7) if background_n else None,
        "new_alerts_in_30m_prestart": int((starts_in_days & prior_30).sum()),
        "new_alerts_in_postrelease_30m": int((starts_in_days & washout & ~active).sum()),
        "observed_minutes_after_warmup_with_score": int((np.isfinite(score) & days).sum()),
        "minutes_after_warmup": int((np.asarray(grid >= grid.floor("D") + pd.Timedelta(minutes=60)) & days).sum()),
    }


def calibrate(scores: dict[str, np.ndarray], grid: pd.DatetimeIndex, eligible: np.ndarray,
              releases: pd.DataFrame, active: np.ndarray, washout: np.ndarray) -> tuple[dict, dict]:
    settings, development = {}, {}
    for key, score in scores.items():
        bkg = score[eligible & np.isfinite(score)]
        if bkg.size == 0:
            raise ValueError(f"No background values for {key}")
        candidates = sorted(set(float(v) for v in np.quantile(bkg, QUANTILES))) + [math.inf]
        chosen = None
        for threshold in candidates:
            metrics = evaluate(score, threshold, grid, eligible, releases, active, washout, DEV)
            if metrics["false_alerts_per_24h"] <= 2 and metrics["background_alarm_occupancy"] <= 0.05:
                chosen, development[key] = threshold, metrics
                break
        assert chosen is not None
        settings[key] = {"threshold_ppm": chosen if math.isfinite(chosen) else None, "disabled_infinite_threshold": not math.isfinite(chosen)}
    def order_single(name: str) -> tuple:
        m = development[name]
        return (-m["timely_30m"], m["false_alerts_per_24h"],
                m["median_detected_delay_minutes"] if m["median_detected_delay_minutes"] is not None else math.inf,
                name)
    best = min((f"single_{name}" for name in NAMES), key=order_single)
    return {"all_candidate_settings": settings, "chosen_single_sensor": best}, development


def threshold_value(settings: dict, name: str) -> float:
    item = settings["all_candidate_settings"][name]
    return math.inf if item["disabled_infinite_threshold"] else float(item["threshold_ppm"])


def selection_phase(source_dir: Path) -> None:
    code_hash = sha256(Path(__file__))
    grid, scores, eligible, releases, active, washout = prepare(source_dir, DEV + SEL)
    # calibrate only on development days, including per-day alarm resets
    dev_days = np.isin(np.asarray(grid.strftime("%Y-%m-%d")), DEV)
    settings, development = calibrate(
        {k: np.where(dev_days, v, np.nan) for k, v in scores.items()},
        grid, eligible & dev_days, releases, active, washout,
    )
    names = [settings["chosen_single_sensor"], "maximum", "two_point_5min"]
    selected = {
        name: evaluate(scores[name], threshold_value(settings, name), grid, eligible, releases, active, washout, SEL)
        for name in names
    }
    if all(selected[n]["timely_30m"] == 0 for n in names):
        preferred = None
    else:
        preferred = min(names, key=lambda name: (
            -selected[name]["timely_30m"], selected[name]["false_alerts_per_24h"],
            selected[name]["median_detected_delay_minutes"]
            if selected[name]["median_detected_delay_minutes"] is not None else math.inf,
            names.index(name),
        ))
    result = {
        "phase": "selection_only",
        "code_sha256": code_hash,
        "source_sha256": SOURCE_SHA,
        "protocol": "docs/stage9-b1-p1-frozen-experiment.md",
        "settings": settings,
        "development_all_eight_singles_and_two_multisensor_rules": development,
        "selection_three_prespecified_rules": selected,
        "selection_daily_metrics": {
            name: {day: evaluate(scores[name], threshold_value(settings, name), grid, eligible, releases,
                                 active, washout, (day,)) for day in SEL}
            for name in names
        },
        "recommended_rule_from_selection": preferred,
        "locked_dates_methane_read": False,
    }
    atomic_new_json(OUT / "p1_selection.json", result)
    print(json.dumps({"chosen_single": settings["chosen_single_sensor"], "recommended": preferred,
                      "selection_metrics": {k: {f: v[f] for f in ("timely_30m", "false_alert_starts", "eligible_background_minutes")} for k, v in selected.items()}}, indent=2))


def locked_phase(source_dir: Path, approved_selection_hash: str) -> None:
    path = OUT / "p1_selection.json"
    if sha256(path) != approved_selection_hash:
        raise ValueError("Selection file differs from frozen digest")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["phase"] != "selection_only" or selection["code_sha256"] != sha256(Path(__file__)):
        raise ValueError("Program changed after selection")
    if selection["locked_dates_methane_read"] is not False:
        raise ValueError("Locked period already read in selection")
    grid, scores, eligible, releases, active, washout = prepare(source_dir, TEST)
    settings = selection["settings"]
    names = [settings["chosen_single_sensor"], "maximum", "two_point_5min"]
    result = {
        "phase": "locked_once",
        "code_sha256": sha256(Path(__file__)),
        "selection_sha256": approved_selection_hash,
        "protocol": selection["protocol"],
        "recommended_rule_from_selection": selection["recommended_rule_from_selection"],
        "settings": {key: settings["all_candidate_settings"][key] for key in names},
        "locked_metrics": {
            name: evaluate(scores[name], threshold_value(settings, name), grid, eligible, releases, active, washout, TEST)
            for name in names
        },
        "locked_daily_metrics": {
            name: {day: evaluate(scores[name], threshold_value(settings, name), grid, eligible, releases,
                                 active, washout, (day,)) for day in TEST}
            for name in names
        },
    }
    atomic_new_json(OUT / "p1_locked_results.json", result)
    print(json.dumps({name: {key: m[key] for key in ("release_events", "timely_30m", "false_alert_starts", "eligible_background_minutes")} for name, m in result["locked_metrics"].items()}, indent=2))


def synthetic_check() -> None:
    t = pd.date_range("2022-05-01 00:00", periods=12, freq="min", tz="UTC")
    s = np.array([np.nan, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, np.nan])
    state, starts = alarms(s, 1, t)
    assert np.flatnonzero(starts).tolist() == [4, 8]
    assert state[0] == -1 and state[5] == 0 and state[11] == -1
    release = pd.DataFrame([{"tc_ExpStartDatetime": t[5] + pd.Timedelta(seconds=10),
                             "tc_ExpEndDatetime": t[9], "tc_EPID": "synthetic"}])
    # Existing alarm from before onset may persist, but only the new alarm at 08 is a match.
    score = np.array([np.nan, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, np.nan])
    truth = evaluate(score, 1, t, np.ones(12, bool), release, np.zeros(12, bool), np.zeros(12, bool), ("2022-05-01",))
    assert truth["timely_30m"] == 1 and truth["event_details"][0]["first_new_alert_utc"] == t[9].isoformat()
    crossing = pd.date_range("2022-05-01 23:57", periods=6, freq="min", tz="UTC")
    _, daily_starts = alarms(np.ones(6), 1, crossing)
    assert np.flatnonzero(daily_starts).tolist() == [2, 5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--phase", choices=("self-test", "selection", "locked"), required=True)
    parser.add_argument("--selection-sha256", help="SHA-256 printed by sha256sum results/stage9_b1/p1_selection.json")
    args = parser.parse_args()
    synthetic_check()
    if args.phase == "self-test":
        print("Synthetic causal alarm and event pairing checks passed; no source data read")
        return
    if args.source_dir is None:
        parser.error("--source-dir is required")
    if args.phase == "selection":
        selection_phase(args.source_dir)
    elif args.phase == "locked":
        if args.selection_sha256 is None:
            parser.error("--selection-sha256 is required for the locked phase")
        locked_phase(args.source_dir, args.selection_sha256)


if __name__ == "__main__":
    main()
