#!/usr/bin/env python3
"""Stage 7A: separate causal lab-gas and smart-building background checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


FROZEN_COMMIT = "ec839cd8959df78ab30eb2ae27c08bf3d7338e83"
FROZEN_SHA = "c0702e7c4f66e30239402657ecf3857d4f358b8f7ac3ae1f19f425c0cb764bb7"
FIELDS = ["time", "target_setpoint", "ethylene_setpoint"] + [f"sensor_{i}" for i in range(16)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_one_hz(archive: zipfile.ZipFile, filename: str, end_s: int) -> dict:
    """Median each completed second; never use gas-delivery columns as predictors."""
    features = np.full((end_s, 16), np.nan)
    counts = np.zeros(end_s, dtype=int)
    both_zero = np.zeros(end_s, dtype=bool)
    last_time = -math.inf
    carry = pd.DataFrame(columns=FIELDS)
    with archive.open(filename) as stream:
        chunks = pd.read_csv(stream, sep=r"\s+", skiprows=1, header=None, names=FIELDS,
                             chunksize=100_000, dtype=float)
        for chunk in chunks:
            if chunk.shape[1] != 19 or not np.isfinite(chunk.time.to_numpy()).all():
                raise ValueError(f"Invalid 322 fields: {filename}")
            times = chunk.time.to_numpy()
            if times[0] < last_time or np.any(np.diff(times) < 0):
                raise ValueError(f"Backwards time: {filename}")
            last_time = times[-1]
            frame = pd.concat((carry, chunk), ignore_index=True) if len(carry) else chunk
            second = np.floor(frame.time.to_numpy()).astype(int)
            latest = second[-1]
            stable = frame.iloc[second < latest]
            carry = frame.iloc[second == latest]
            if len(stable):
                fill_bins(stable, features, counts, both_zero, end_s)
    if len(carry):
        fill_bins(carry, features, counts, both_zero, end_s)
    return {"features": features, "valid": np.isfinite(features).all(axis=1),
            "counts": counts, "both_zero": both_zero}


def fill_bins(frame, features, counts, both_zero, end_s):
    groups = frame.groupby(np.floor(frame.time.to_numpy()).astype(int), sort=True)
    rows = groups[FIELDS[3:]].median()
    bins = rows.index.to_numpy(dtype=int)
    include = (bins >= 0) & (bins < end_s)
    bins = bins[include]
    features[bins] = rows.to_numpy()[include]
    counts[bins] = groups.size().to_numpy()[include]
    max_values = groups[FIELDS[1:3]].max().to_numpy()[include]
    min_values = groups[FIELDS[1:3]].min().to_numpy()[include]
    both_zero[bins] = np.all(max_values == 0, axis=1) & np.all(min_values == 0, axis=1)


def timeslice_mask(runs, interval, guard_s, size):
    start, end = interval
    mask = np.zeros(size, dtype=bool)
    for run in runs:
        if run["state"] != "both_zero":
            continue
        from_s = max(run["start_s"], start) + guard_s
        until_s = min(run["end_s"], end)
        # Only complete [n,n+1) bins entirely contained in this zero-setpoint run.
        first_bin = max(start, math.ceil(from_s - 1e-9))
        last_bin = min(end - 1, math.floor(until_s - 1 + 1e-9))
        if first_bin <= last_bin:
            mask[first_bin:last_bin + 1] = True
    return mask


def eligible_onsets(runs, interval, prior_s, dwell_s, horizon_s):
    start, end = interval
    result = []
    for before, after in zip(runs, runs[1:]):
        if (before["state"] == "both_zero" and after["state"] == "target_present"
                and before["start_s"] >= start and before["duration_s"] >= prior_s
                and after["duration_s"] >= dwell_s
                and start <= after["start_s"] and after["start_s"] + horizon_s <= end):
            result.append(after["start_s"])
    return result


def score_sensors(values, valid, center, scale, method, selected_channel):
    absz = np.abs((values - center) / scale)
    score = absz[:, selected_channel] if method == "B0" else np.median(absz, axis=1)
    return np.where(valid, score, np.nan)


def transitions(score, threshold, interval, persistence):
    """Return first transition times and the start of their G1 evidence, never oracle-reset."""
    start, end = interval
    result = []
    active = False
    consecutive = 0
    origin = None
    for n in range(start, end):
        high = bool(np.isfinite(score[n]) and score[n] > threshold)
        if high:
            if not consecutive:
                origin = n + 1
            consecutive += 1
        else:
            consecutive = 0
            active = False
            origin = None
        now = consecutive >= persistence
        if now and not active:
            result.append({"time_s": n + 1, "evidence_start_s": origin})
        active = now
    return result


def evaluate_uci(filename, series, audit, config, output):
    settings = config["uci322"]
    runs = audit[filename]["runs"]
    size = len(series["features"])
    blocks = {name: tuple(x) for name, x in settings["intervals_s"].items()}
    masks = {name: timeslice_mask(runs, span, settings["negative_both_zero_min_prior_s"], size)
             for name, span in blocks.items()}
    training_mask = timeslice_mask(runs, blocks["train"],
                                   settings["development_baseline_both_zero_min_prior_s"], size)
    training_mask &= series["both_zero"] & series["valid"]
    if np.count_nonzero(training_mask) < 300:
        raise ValueError(f"Fewer than 300 clean calibration seconds for {filename}")
    train = series["features"][training_mask]
    center = np.median(train, axis=0)
    scale = 1.4826 * np.median(np.abs(train - center), axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"Zero or invalid MAD in {filename}; stop, do not tune fallback")
    qualified = {name: eligible_onsets(runs, span,
                                       settings["positive_onset_prior_both_zero_s"],
                                       settings["positive_target_dwell_min_s"],
                                       settings["event_scoring_horizon_s"])
                 for name, span in blocks.items()}
    if any(not qualified[name] for name in ("train", "validation")):
        raise ValueError(f"No train or validation event in {filename}")
    z = np.abs((series["features"] - center) / scale)
    train_background = np.median(z[training_mask], axis=0)
    contrasts = []
    for ch in range(16):
        response = []
        for t0 in qualified["train"]:
            left, right = math.ceil(t0), math.ceil(t0 + settings["event_scoring_horizon_s"])
            finite = z[left:right, ch][np.isfinite(z[left:right, ch])]
            if len(finite):
                response.append(float(np.median(finite)))
        contrasts.append(float(np.median(response) - train_background[ch]))
    selected_channel = int(np.argmax(contrasts))  # argmax takes lowest channel on tie
    result = []
    event_rows = []
    for method in ("B0", "B1"):
        signal = score_sensors(series["features"], series["valid"], center, scale,
                               method, selected_channel)
        for q in settings["threshold_quantiles_training_only"]:
            cutoff = float(np.quantile(signal[training_mask], q, method="linear"))
            for name, span in blocks.items():
                events = transitions(signal, cutoff, span, settings["evidence_consecutive_bins"])
                valid_negative = masks[name] & series["both_zero"] & series["valid"]
                negative_count = sum(valid_negative[math.ceil(e["time_s"]) - 1]
                                     for e in events)
                negative_active = sum(valid_negative[n] and signal[n] > cutoff
                                      for n in range(span[0], span[1]))
                detected = 0
                for t0 in qualified[name]:
                    matching = [e for e in events if t0 < e["evidence_start_s"]
                                and e["time_s"] <= t0 + settings["event_scoring_horizon_s"]]
                    first = matching[0] if matching else None
                    prior = [e for e in events if e["evidence_start_s"] <= t0
                             and t0 < e["time_s"] <= t0 + settings["event_scoring_horizon_s"]]
                    # Already-active alarm does not trigger another transition.
                    preexisting = any(e["time_s"] <= t0 and
                                      not any(t0 >= n + 1 and
                                              (not np.isfinite(signal[n]) or signal[n] <= cutoff)
                                              for n in range(e["time_s"], math.floor(t0)))
                                      for e in events)
                    detected += bool(first)
                    event_rows.append({"file": filename, "split": name, "method": method,
                                       "q": q, "setpoint_onset_s": t0,
                                       "first_g2_s": first["time_s"] if first else "",
                                       "delay_s": round(first["time_s"] - t0, 2) if first else "",
                                       "preexisting_evidence_events": len(prior),
                                       "already_active_at_onset": int(preexisting)})
                duration_h = float(np.sum(valid_negative)) / 3600
                result.append({"file": filename, "split": name, "method": method,
                               "quantile": q, "threshold": round(cutoff, 8),
                               "qualified_onsets": len(qualified[name]), "detected_within_60s": detected,
                               "negative_hours": round(duration_h, 6),
                               "new_negative_alarm_events": negative_count,
                               "negative_alarm_events_per_hour":
                                   round(negative_count / duration_h, 6) if duration_h else "",
                               "negative_high_score_seconds": negative_active,
                               "available_sensor_seconds": int(np.sum(series["valid"][span[0]:span[1]])),
                               "score_seconds": span[1] - span[0],
                               "selected_channel_0_based": selected_channel if method == "B0" else ""})
    # Offline time-only sham: learn one phase at a 200 s cycle from training setpoints.
    period = 200
    # A completed one-second window cannot fire before a setpoint change at t=.01.
    phase_counts = np.bincount([math.ceil(t) % period for t in qualified["train"]], minlength=period)
    clock_phase = int(np.argmax(phase_counts))
    clock_rows = []
    for name, span in blocks.items():
        fake = np.arange(span[0] + 1, span[1] + 1)
        pulses = fake[fake % period == clock_phase]
        negative = masks[name] & series["both_zero"] & series["valid"]
        clock_rows.append({"file": filename, "split": name, "period_s": period,
                           "phase_s": clock_phase, "label_oracle_warning": "not_a_sensor_detector",
                           "qualified_onsets": len(qualified[name]),
                           "events_hit_within_60s": sum(np.any((pulses > t) & (pulses <= t + 60))
                                                        for t in qualified[name]),
                           "negative_pulses": sum(negative[t - 1] for t in pulses)})
    csv_write(output / f"uci322_{filename.split('.')[0]}_events.csv", event_rows,
              list(event_rows[0]))
    return result, clock_rows, {"file": filename, "selected_channel_0_based": selected_channel,
                                "train_negative_s_guard60": int(np.sum(training_mask)),
                                "train_channel_center": center.tolist(),
                                "train_channel_mad_scale": scale.tolist(),
                                "train_channel_contrasts": contrasts}


def load_sb112(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    result = {}
    for sheet in workbook.worksheets:
        values = iter(sheet.values)
        header = next(values)
        if header != ("time", "data_type", "unit", "data_value"):
            raise ValueError(f"Unexpected SB112 schema in {path}/{sheet.title}")
        rows = []
        for raw_time, gas, unit, value in values:
            if raw_time is None:
                continue
            when = datetime.fromisoformat(raw_time)
            reading = float(value)
            if not np.isfinite(reading) or unit != "ppm":
                raise ValueError(f"Invalid sample/unit in {path}/{sheet.title}")
            rows.append((when, reading))
        if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"SB112 missing/non-increasing times: {path}/{sheet.title}")
        result[rows[0][0].date().isoformat()] = rows
    workbook.close()
    return result


def sb112_day(rows, threshold, persistence, max_gap, merge_s):
    covered = 0.0
    segment = 0
    consecutive = 0
    alarm = False
    last_clear = None
    merged = []
    triggers = []
    prev = None
    for when, reading in rows:
        delta = (when - prev).total_seconds() if prev else None
        if delta is not None and (delta <= 0 or delta > max_gap):
            segment += 1
            consecutive = 0
            alarm = False
            last_clear = None
        elif delta is not None:
            covered += delta
        high = reading > threshold
        consecutive = consecutive + 1 if high else 0
        now = consecutive >= persistence
        if not now and alarm:
            last_clear = when
        if now and not alarm:
            triggers.append((when, segment))
            if not (merged and merged[-1][1] == segment and last_clear is not None
                    and (when - last_clear).total_seconds() < merge_s):
                merged.append((when, segment))
        alarm = now
        prev = when
    return {"coverage_h": covered / 3600, "events": len(merged),
            "raw_transitions": len(triggers), "first": merged[0][0].isoformat() if merged else "",
            "episode_times": [x[0].isoformat() for x in merged],
            "segments": segment + 1}


def sb112(config, root, output):
    settings = config["sb112"]
    results = []
    episode_rows = []
    runinfo = []
    for file in settings["sensors_separate"]:
        path = root / file
        per_day = load_sb112(path)
        assignment = settings["day_assignment_local_plus_02"]
        if sorted(per_day) != sorted(assignment.values()):
            raise ValueError(f"Date mismatch in {file}")
        train = np.array([v for _, v in per_day[assignment["train"]]], dtype=float)
        if len(train) < 100:
            raise ValueError(f"Insufficient training day values: {file}")
        thresholds = {"primary": float(np.quantile(train, settings["primary_quantile_train_only"],
                                                    method="linear")),
                      "sensitivity": float(np.quantile(train,
                                                       settings["sensitivity_quantile_train_only"],
                                                       method="linear"))}
        for variant, cutoff in thresholds.items():
            persistence = settings["primary_consecutive_observations"] if variant == "primary" else settings["sensitivity_consecutive_observations"]
            for split in ("validation", "test"):
                rows = per_day[assignment[split]]
                scored = sb112_day(rows, cutoff, persistence,
                                   settings["gap_breaks_continuity_if_over_s"],
                                   settings["episode_merge_clear_time_s"])
                results.append({"sensor_file": file, "day": assignment[split], "split": split,
                                "variant": variant, "training_threshold_ppm": round(cutoff, 8),
                                "consecutive_observations": persistence,
                                "sample_count": len(rows), "day_median_ppm": float(np.median([r[1] for r in rows])),
                                "coverage_h": round(scored["coverage_h"], 6),
                                "events": scored["events"],
                                "events_per_covered_hour": round(scored["events"] / scored["coverage_h"], 6) if scored["coverage_h"] else "",
                                "raw_transitions": scored["raw_transitions"],
                                "first_event_local": scored["first"],
                                "coverage_segments": scored["segments"]})
                for onset in scored["episode_times"]:
                    episode_rows.append({"sensor_file": file, "split": split,
                                         "variant": variant, "event_start_local": onset})
        stimulus = per_day[assignment["qualitative_only"]]
        runinfo.append({"sensor_file": file, "file_sha256": sha256(path),
                        "training_day_median_ppm": float(np.median(train)),
                        "training_day_sample_count": len(train),
                        "stimulus_day_samples_without_onset_labels": len(stimulus),
                        "stimulus_day_min_ppm": min(v for _, v in stimulus),
                        "stimulus_day_max_ppm": max(v for _, v in stimulus),
                        "thresholds_ppm": thresholds})
    csv_write(output / "sb112_episodes.csv", episode_rows,
              ["sensor_file", "split", "variant", "event_start_local"])
    csv_write(output / "sb112_summary.csv", results, list(results[0]))
    return runinfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage7_a_protocol_candidate.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/uci_322_original.zip"))
    parser.add_argument("--audit", type=Path, default=Path("results/stage7_e1/uci322_source_audit.json"))
    parser.add_argument("--sb112", type=Path, default=Path("data/raw/sb112_zenodo_6616632"))
    parser.add_argument("--out", type=Path, default=Path("results/stage7_a"))
    args = parser.parse_args()
    if sha256(args.config) != FROZEN_SHA:
        raise ValueError("Frozen config checksum mismatch; stop before evaluation")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "approved_frozen_before_model_results":
        raise ValueError("Protocol is not approved")
    source_hash = sha256(args.archive)
    if source_hash != config["uci322"]["archive_sha256"]:
        raise ValueError("Wrong UCI322 ZIP")
    source_audit = json.loads(args.audit.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, clock_rows, parameters = [], [], []
    with zipfile.ZipFile(args.archive) as archive:
        for filename in config["uci322"]["files_independent"]:
            series = load_one_hz(archive, filename, config["uci322"]["intervals_s"]["test"][1])
            data, time_control, params = evaluate_uci(filename, series, source_audit, config, args.out)
            all_rows.extend(data)
            clock_rows.extend(time_control)
            parameters.append(params)
    csv_write(args.out / "uci322_summary.csv", all_rows, list(all_rows[0]))
    csv_write(args.out / "uci322_clock_only_diagnostic.csv", clock_rows, list(clock_rows[0]))
    sb_info = sb112(config, args.sb112, args.out)
    run = {"status": "completed", "source_safety_note":
           "UCI target=gas-delivery setpoint; SB112=no-manual-trigger background; neither is coke-plant accident truth",
           "frozen_commit": FROZEN_COMMIT, "frozen_config_sha256": FROZEN_SHA,
           "source_uci322_sha256": source_hash, "audit_uci322_sha256": sha256(args.audit),
           "script_sha256": sha256(Path(__file__)),
           "python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "openpyxl": openpyxl.__version__,
           "gas_training_parameters": parameters, "sb112_files": sb_info}
    (args.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Stage 7A complete:", len(all_rows), "UCI rows;", len(sb_info), "separate SB112 channels")


if __name__ == "__main__":
    main()
