#!/usr/bin/env python3
"""UCI 309: causal laboratory gas-change evidence, with file-level validation.

The release schedule is used only in offline scoring. All metrics refer to the
wind-tunnel experiment, never an industrial accident or calibrated alarm.

python scripts/run_m1_uci309.py --archive ../uci_309_original.zip \
    --out results/m1_uci309
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260924
NAME = re.compile(r"^(\d{3})_Et_([nLMH])_(CO|Me)_([nLMH])$")
COLS = ["time_s", "temp_c", "rh_pct"] + [f"sensor_{i}_adc" for i in range(1, 9)]
QUANTILES = (0.95, 0.99, 0.995, 0.999)
METHODS = ("single", "multi", "ewma", "cusum")


@dataclass
class Trial:
    name: str
    config: str
    # Array indices are second-long windows [n,n+1); outputs are available at n+1.
    gas: np.ndarray
    ambient: np.ndarray
    gas_valid: np.ndarray
    ambient_valid: np.ndarray
    split: str = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trials(archive: Path) -> list[Trial]:
    trials: list[Trial] = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(x for x in z.namelist() if x.startswith("dataset_twosources_downsampled/") and not x.endswith("/"))
        if len(names) != 180:
            raise ValueError(f"Expected 180 independent downsampled trials, got {len(names)}")
        for name in names:
            match = NAME.fullmatch(name.rsplit("/", 1)[-1])
            if not match:
                raise ValueError(f"Unexpected trial name: {name}")
            frame = pd.read_csv(z.open(name), header=None, names=COLS)
            if frame.shape[1] != 11:
                raise ValueError(f"Invalid column count for {name}")
            raw_time = frame.time_s.to_numpy(dtype=float)
            if not np.isfinite(frame.to_numpy(dtype=float)).all() or np.any(np.diff(raw_time) < 0):
                raise ValueError(f"Nonfinite values or backward time in {name}")
            seconds = np.floor(raw_time).astype(int)
            if (seconds < 0).any():
                raise ValueError(f"Negative time in {name}")
            # A median for each completed second preserves all duplicate readings.
            frame = frame.assign(second=seconds)
            medians = frame.groupby("second", sort=True)[COLS[1:]].median()
            bins = medians.reindex(range(240))
            gas = bins[COLS[3:]].to_numpy(dtype=float)
            ambient = bins[COLS[1:3]].to_numpy(dtype=float)
            gas_valid = np.isfinite(gas).all(axis=1) & ((gas > 0) & (gas < 3110)).all(axis=1)
            ambient_valid = np.isfinite(ambient).all(axis=1)
            file_id, eth, species, level = match.groups()
            trials.append(Trial(file_id + "_Et_" + eth + "_" + species + "_" + level,
                                eth + "/" + species + "/" + level,
                                gas, ambient, gas_valid, ambient_valid))
    return trials


def assign_splits(trials: list[Trial]) -> list[dict]:
    groups: dict[str, list[Trial]] = {}
    for trial in trials:
        groups.setdefault(trial.config, []).append(trial)
    if len(groups) != 30 or any(len(g) != 6 for g in groups.values()):
        raise ValueError("Expected 30 configurations with six independent trials each")
    rng = np.random.default_rng(SEED)
    manifest = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda x: x.name)
        for i, index in enumerate(rng.permutation(6)):
            trial = members[int(index)]
            trial.split = "train" if i < 3 else ("validation" if i == 3 else "test")
            manifest.append({"trial": trial.name, "configuration": key, "split": trial.split})
    assert [sum(t.split == split for t in trials) for split in ("train", "validation", "test")] == [90, 30, 60]
    return sorted(manifest, key=lambda x: x["trial"])


def channels(trial: Trial, source: str) -> tuple[np.ndarray, np.ndarray]:
    return (trial.gas, trial.gas_valid) if source == "gas" else (trial.ambient, trial.ambient_valid)


def calibration(trials: list[Trial], source: str, mode: str) -> dict:
    train = [t for t in trials if t.split == "train"]
    if mode not in ("A", "B"):
        raise ValueError(mode)
    centers = np.array([np.median(channels(t, source)[0][5:25], axis=0) for t in train])
    if any(not channels(t, source)[1][5:25].all() for t in train):
        raise ValueError("Missing calibration seconds in a training trial")
    global_center = np.median(centers, axis=0)
    background = np.concatenate([
        channels(t, source)[0][25:60] -
        (np.median(channels(t, source)[0][5:25], axis=0) if mode == "A" else global_center)
        for t in train
    ])
    if not np.isfinite(background).all():
        raise ValueError("Invalid training background; do not silently impute")
    median_residual = np.median(background, axis=0)
    mad = 1.4826 * np.median(np.abs(background - median_residual), axis=0)
    scale = np.maximum(mad, np.maximum(np.abs(global_center) * 0.001, 0.001))
    return {"center": global_center, "scale": scale, "mode": mode, "source": source}


def normalize(trial: Trial, cal: dict, frozen: bool = False) -> tuple[np.ndarray, np.ndarray]:
    data, valid = channels(trial, cal["source"])
    if cal["mode"] == "A":
        if not valid[5:25].all():
            return np.full_like(data, np.nan), np.zeros(len(valid), dtype=bool)
        center = np.median(data[5:25], axis=0)
    else:
        center = cal["center"]
    # Frozen-gas negative control is synthesized *after* training parameters are frozen.
    residual = np.zeros_like(data) if frozen else (data - center) / cal["scale"]
    return residual, valid.copy()


def detector_series(trial: Trial, cal: dict, method: str, sensor: int | None = None,
                    frozen: bool = False) -> tuple[np.ndarray, np.ndarray]:
    z, valid = normalize(trial, cal, frozen=frozen)
    raw = np.abs(z)
    if method == "single":
        assert sensor is not None
        score = raw[:, sensor]
    else:
        # Correlated channels are aggregated, never counted as independent alarms.
        score = np.median(raw, axis=1)
    if method in ("single", "multi"):
        return score, valid
    result = np.full(len(score), np.nan)
    if method == "ewma":
        state = 0.0
        for second in range(25, 240):
            if valid[second]:
                state = 0.3 * score[second] + 0.7 * state
                result[second] = state
            else:
                state = 0.0  # interruption breaks evidence; cannot carry across missing time
    elif method == "cusum":
        state = 0.0
        reference = cal["cusum_reference"]
        for second in range(25, 240):
            if valid[second]:
                state = max(0.0, state + score[second] - reference)
                result[second] = state
            else:
                state = 0.0
    else:
        raise ValueError(method)
    return result, valid


def fit_cusum_reference(trials: list[Trial], cal: dict) -> None:
    bg = np.concatenate([detector_series(t, cal, "multi")[0][25:60]
                         for t in trials if t.split == "train"])
    bg = bg[np.isfinite(bg)]
    median = float(np.median(bg))
    robust_std = 1.4826 * float(np.median(np.abs(bg - median)))
    cal["cusum_reference"] = median + 0.5 * max(robust_std, 0.001)


def evidence_events(score: np.ndarray, valid: np.ndarray, threshold: float,
                    duration_s: int = 3) -> dict:
    """Run without reading release metadata; event times refer to completed seconds."""
    previous_g1 = previous_g2 = False
    consecutive = 0
    g1_events: list[int] = []
    g2_events: list[int] = []
    g2_origins: list[int] = []
    run_start: int | None = None
    g2_at_sixty = False
    missing = 0
    for second in range(25, 240):
        if valid[second] and np.isfinite(score[second]):
            g1 = bool(score[second] > threshold)
            consecutive = consecutive + 1 if g1 else 0
            g2 = consecutive >= duration_s
        else:
            missing += 1
            g1 = g2 = False
            consecutive = 0
        # second+1 is the first time this complete [second,second+1) window is usable.
        when = second + 1
        if g1 and not previous_g1:
            g1_events.append(when)
            run_start = when
        if g2 and not previous_g2:
            g2_events.append(when)
            assert run_start is not None
            g2_origins.append(run_start)
        if when == 60:
            g2_at_sixty = g2
        previous_g1, previous_g2 = g1, g2
    return {"g1_events": g1_events, "g2_events": g2_events, "g2_origins": g2_origins,
            "g2_at_sixty": g2_at_sixty, "invalid_seconds": missing}


def score_events(events: dict) -> dict:
    before = [t for t in events["g2_events"] if 25 < t <= 60]
    # An alert completed after 60 s but based on an already abnormal pre-release
    # run is not a response newly attributable to that release.
    seeded = [t for t, start in zip(events["g2_events"], events["g2_origins"])
              if 60 < t <= 240 and start <= 60]
    after = [t for t, start in zip(events["g2_events"], events["g2_origins"])
             if 60 < t <= 240 and start > 60]
    first = after[0] if after else None
    return {"false_events": len(before), "false_file": int(bool(before)),
            "pre_release_seeded_events": len(seeded),
            "detected": int(first is not None), "detected_within_60_s": int(first is not None and first <= 120),
            "first_detection_s": first, "delay_s": first - 60 if first is not None else None,
            "already_g2_at_release": int(events["g2_at_sixty"]),
            "invalid_seconds": events["invalid_seconds"],
            "g1_before_count": sum(25 < t <= 60 for t in events["g1_events"]),
            "g1_after_count": sum(60 < t <= 240 for t in events["g1_events"])}


def evaluate(trials: list[Trial], cal: dict, method: str, threshold: float,
             split: str, sensor: int | None = None, frozen: bool = False,
             duration_s: int = 3) -> list[dict]:
    rows = []
    for trial in trials:
        if trial.split != split:
            continue
        series, valid = detector_series(trial, cal, method, sensor, frozen)
        metrics = score_events(evidence_events(series, valid, threshold, duration_s))
        rows.append({"trial": trial.name, "configuration": trial.config,
                     "split": split, "mode": cal["mode"], "source": cal["source"],
                     "method": method, "sensor": sensor, "threshold": threshold,
                     "control": "synthetic_frozen_gas" if frozen else "real_laboratory_data",
                     **metrics})
    return rows


def summarize(rows: list[dict]) -> dict:
    delays = [r["delay_s"] for r in rows if r["delay_s"] is not None]
    return {"n_files": len(rows), "detected": sum(r["detected"] for r in rows),
            "detected_within_60_s": sum(r["detected_within_60_s"] for r in rows),
            "false_files": sum(r["false_file"] for r in rows),
            "false_events": sum(r["false_events"] for r in rows),
            "background_seconds": 35 * len(rows),
            "delay_median_detected_s": float(np.median(delays)) if delays else None,
            "delay_p90_detected_s": float(np.percentile(delays, 90)) if delays else None,
            "already_g2_at_release": sum(r["already_g2_at_release"] for r in rows),
            "pre_release_seeded_events": sum(r["pre_release_seeded_events"] for r in rows),
            "invalid_monitor_seconds": sum(r["invalid_seconds"] for r in rows)}


def choose_on_validation(candidates: list[dict]) -> dict:
    """Precommitted ranking: <=1 false file, early detections, all detections, capped delay."""
    def key(candidate: dict) -> tuple:
        s = candidate["summary"]
        delay_penalty = sum(r["delay_s"] if r["delay_s"] is not None else 180
                            for r in candidate["rows"])
        return (s["false_files"] <= 1, -s["false_files"],
                s["detected_within_60_s"], s["detected"], -delay_penalty,
                candidate["quantile"], -(candidate["sensor"] or 0))
    return max(candidates, key=key)


def candidate_thresholds(trials: list[Trial], cal: dict, method: str,
                         sensor: int | None = None) -> list[tuple[float, float]]:
    bg = np.concatenate([detector_series(t, cal, method, sensor)[0][25:60]
                         for t in trials if t.split == "train"])
    bg = bg[np.isfinite(bg)]
    return [(q, float(np.quantile(bg, q))) for q in QUANTILES]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/m1_uci309"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trials = load_trials(args.archive)
    manifest = assign_splits(trials)
    write_csv(args.out / "split_manifest.csv", manifest)

    selections = []
    files = []
    summaries = []
    selected_parameters = []
    duration_sensitivity = []
    for mode in ("A", "B"):
        for source in ("gas", "ambient"):
            if source == "ambient" and mode != "A":
                continue  # real measured environmental control, one protocol only
            cal = calibration(trials, source, mode)
            fit_cusum_reference(trials, cal)
            method_names = METHODS if source == "gas" else ("multi", "ewma", "cusum")
            for method in method_names:
                candidates = []
                sensors = range(8) if method == "single" else (None,)
                for sensor in sensors:
                    for quantile, threshold in candidate_thresholds(trials, cal, method, sensor):
                        rows = evaluate(trials, cal, method, threshold, "validation", sensor)
                        candidates.append({"rows": rows, "summary": summarize(rows),
                                           "sensor": sensor, "quantile": quantile,
                                           "threshold": threshold})
                best = choose_on_validation(candidates)
                for candidate in candidates:
                    selections.append({"mode": mode, "source": source, "method": method,
                                       "sensor": candidate["sensor"], "quantile": candidate["quantile"],
                                       "threshold": candidate["threshold"],
                                       "selected": int(candidate is best), **candidate["summary"]})
                selected_parameters.append({"mode": mode, "source": source, "method": method,
                                            "sensor": best["sensor"], "quantile": best["quantile"],
                                            "threshold": best["threshold"],
                                            "global_center": cal["center"].tolist(),
                                            "scale": cal["scale"].tolist(),
                                            "cusum_reference": cal["cusum_reference"]})
                for split in ("validation", "test"):
                    rows = evaluate(trials, cal, method, best["threshold"], split, best["sensor"])
                    files.extend(rows)
                    summaries.append({"mode": mode, "source": source, "method": method,
                                      "split": split, "control": "real_laboratory_data", **summarize(rows)})
                if source == "gas":
                    null = evaluate(trials, cal, method, best["threshold"], "test", best["sensor"], True)
                    files.extend(null)
                    summaries.append({"mode": mode, "source": source, "method": method,
                                      "split": "test", "control": "synthetic_frozen_gas", **summarize(null)})
                    # Exploratory sensitivity on the sealed model: never reselect thresholds.
                    for duration in (1, 3, 5):
                        sensitivity = evaluate(trials, cal, method, best["threshold"], "test",
                                               best["sensor"], duration_s=duration)
                        duration_sensitivity.append({"mode": mode, "source": source, "method": method,
                                                     "duration_s": duration, **summarize(sensitivity)})
    write_csv(args.out / "selection_validation.csv", selections)
    write_csv(args.out / "file_results.csv", files)
    write_csv(args.out / "summary.csv", summaries)
    write_csv(args.out / "duration_sensitivity.csv", duration_sensitivity)
    provenance = {
        "run_id": "m1-uci309-20260924-v2", "data_kind": "UCI_309_laboratory",
        "validation_status": "exploratory_pilot_repeated_holdout_after_scoring_fix",
        "archive_sha256": sha256(args.archive), "script_sha256": sha256(Path(__file__)),
        "split_manifest_sha256": sha256(args.out / "split_manifest.csv"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
        "seed": SEED, "source_bins": "[n,n+1) median, emitted at n+1",
        "baseline_A_s": [5, 25], "background_evaluation_s": [25, 60],
        "release_program_s": [60, 240], "consecutive_valid_seconds_for_G2": 3,
        "release_boundary_policy": "G2 evidence seeded before release is reported separately, not credited to release detection",
        "duration_sensitivity_seconds": [1, 3, 5],
        "quantiles_train_background": list(QUANTILES),
        "validation_ranking": "<=1 false file; fewer false files; more within-60s detections; more total detections; less capped delay; quantile",
        "clock_oracle": {"meaning": "program-label leakage demonstration only",
                         "test_n_files": 60, "detected": 60, "false_files": 0,
                         "idealized_delay_s": 0},
        "frozen_gas_note": "Synthetic constant-baseline input; not a measured negative trial",
        "conclusion_scope": "Known-clean-start laboratory detection, no site safety validation",
        "parameters": selected_parameters,
    }
    (args.out / "run.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "test_summary": [row for row in summaries
                       if row["split"] == "test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
