#!/usr/bin/env python3
"""Freeze metadata-only date split for the proposed UCI 362 experiment.

This does not inspect sensor values or run models. The source audit is required.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


AUDIT = Path("results/stage9_uci362/source_audit.json")
OUTPUT = Path("results/stage9_uci362/proposed_split_manifest.json")
SEED = "20260926"


def rank(date: str, stratum: str) -> str:
    return hashlib.sha256(f"{SEED}|{stratum}|{date}".encode()).hexdigest()


def main() -> None:
    source = json.loads(AUDIT.read_text(encoding="utf-8"))
    if source["sha256"] != "7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395":
        raise ValueError("Unexpected source archive")
    by_date: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for key, trial in source["trials"].items():
        by_date[trial["date"]].append((int(key), trial["class"]))

    strata: dict[str, list[str]] = defaultdict(list)
    for date, members in by_date.items():
        classes = {c for _, c in members}
        stratum = ("background_only" if classes == {"background"}
                   else ("mixed" if "background" in classes else "stimulus_only"))
        strata[stratum].append(date)
    if {k: len(v) for k, v in strata.items()} != {"background_only": 10, "mixed": 20, "stimulus_only": 28}:
        raise ValueError("Unexpected metadata strata")

    assignments = {}
    for stratum, dates in sorted(strata.items()):
        ordered = sorted(dates, key=lambda date: rank(date, stratum))
        train_count = {"background_only": 6, "mixed": 12, "stimulus_only": 16}[stratum]
        validation_count = {"background_only": 2, "mixed": 4, "stimulus_only": 6}[stratum]
        for n, date in enumerate(ordered):
            assignments[date] = "train" if n < train_count else ("validation" if n < train_count + validation_count else "test")

    trials = [
        {"id": ident, "date": date, "class": label, "split": assignments[date],
         "calibration_bins_15min": source["trials"][str(ident)]["calibration_bins_15min"],
         "pre_background_bins_45min": source["trials"][str(ident)]["pre_background_bins_45min"],
         "post_stimulus_bins_5min": source["trials"][str(ident)]["post_stimulus_bins_5min"]}
        for date in sorted(by_date)
        for ident, label in sorted(by_date[date])
    ]
    summary = {}
    for split in ("train", "validation", "test"):
        selected = [r for r in trials if r["split"] == split]
        summary[split] = {"days": sum(s == split for s in assignments.values()),
                          "series": len(selected),
                          "classes": dict(sorted(Counter(r["class"] for r in selected).items())),
                          "calibration_eligible": sum(r["calibration_bins_15min"] >= 600 for r in selected),
                          "positive_eligible": sum(r["class"] != "background" and r["calibration_bins_15min"] >= 600
                                                   and r["post_stimulus_bins_5min"] >= 120 for r in selected)}
    result = {"source_sha256": source["sha256"], "seed": SEED,
              "policy": "whole-date split; SHA-256 rank within background_only/mixed/stimulus_only strata; 6/2/2, 12/4/4, 16/6/6 days",
              "summary": summary, "trials": trials}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
