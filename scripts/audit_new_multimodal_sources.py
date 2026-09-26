#!/usr/bin/env python3
"""Read-only source/availability audit for newly found industrial datasets.

The RoboFusion spreadsheet is a partial public sample. Counts below describe
the file, not independent hazards, actual gas species concentrations, or model
performance. Raw media and spreadsheets remain ignored under data/raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "09c32cc594e7ae74e62e4e583623e320e9ed04e186f0987c6d5638e6cba0a428"
DEFAULT_INPUT = ROOT / "data/raw/robofusion_12d/Real_Dataset_Normal_Hazard_12 Days_ 3 Hazards.xlsx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(path: Path) -> dict:
    if sha256(path) != EXPECTED:
        raise ValueError("RoboFusion source sample differs from audited publisher file")
    df = pd.read_excel(path, sheet_name="Sheet1")
    required = {"Date", "Time", "Suite_ID", "Condition", "Status"}
    if not required.issubset(df):
        raise ValueError(f"Missing mandatory columns: {required - set(df)}")
    stamp = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    df["audit_timestamp"] = stamp
    conditions = {}
    for label, group in df.groupby("Condition"):
        by_suite = group.groupby("Suite_ID").size().to_dict()
        conditions[label] = {
            "rows": len(group),
            "unique_minutes": int(group.audit_timestamp.nunique()),
            "dates": sorted(set(group["Date"].astype(str))),
            "rows_by_suite": {str(k): int(v) for k, v in by_suite.items()},
            "first_timestamp": str(group.audit_timestamp.min()),
            "last_timestamp": str(group.audit_timestamp.max()),
        }
    return {
        "status": "source_audit_only; not predictive, not an accident validation",
        "source": "https://github.com/Amr-Khamis-84/RoboFusion-Dataset",
        "source_commit": "d3907eda585cf202fc00c462b550d2512d5b9e6f",
        "raw_filename": path.name,
        "sha256": EXPECTED,
        "bytes": path.stat().st_size,
        "rows": int(len(df)),
        "columns": [c for c in df if c != "audit_timestamp"],
        "dates": [str(df.Date.min()), str(df.Date.max())],
        "suites": {str(k): int(v) for k, v in df.Suite_ID.value_counts().items()},
        "status_original_or_interpolation": {str(k): int(v) for k, v in df.Status.value_counts().items()},
        "duplicate_timestamp_suite": int(df.duplicated(["audit_timestamp", "Suite_ID"]).sum()),
        "conditions": conditions,
        "rows_missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items() if v},
        "scope": "Public 12-day file has no video, personnel coordinates, operator release log, or per-suite positions; MQ channels are sensor readings and cannot be asserted calibrated species concentrations from this spreadsheet alone.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage9_new_sources/robofusion_12d_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": result["rows"], "conditions": result["conditions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
