#!/usr/bin/env python3
"""Describe downloaded E1 source files; no model training or event labelling."""

import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import openpyxl


TADI = {
    "Logger_A.csv": "43f75010a191045df5b62c04d94bdc6e",
    "Logger_C.csv": "3f41c17015930af0f96e27313ce10217",
    "Logger_D.csv": "f2842db9627cfc7a8b1d72c6877bfb7c",
    "Logger_E.csv": "1763c7ff7034b0d75bd0e8c458eabb1e",
    "Logger_F.csv": "b58620734abcb083bf2cdd7b6e70654e",
    "Logger_H.csv": "a1d74539a4d538b84ec9e3e0217d0950",
    "Loggers_COLUMNS.txt": "4463e2140e390081745d97cf4ef56656",
    "sonic3d_anemometer.csv": "4c5fa615ca97bd74ebd2ad7dd51b94f3",
    "sonic3d_anemometer_COLUMNS.txt": "03770a4acd8b31727cf795fefbd2e854",
}
SB112 = {
    "cng_sensor.xlsx": "58ed493edd5420a850b580aa85cac985",
    "co_sensor.xlsx": "56f0fc96c5135f411dafc3513d47ac20",
    "lpg_sensor.xlsx": "544b905e311e6fc74f053c76324a1c9a",
    "smoke_sensor.xlsx": "068af1a5176193d4a1315ba002a5d58a",
}


def checked(path, expected):
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Source checksum mismatch: {path}: {actual}")
    return {"file": path.name, "size_bytes": path.stat().st_size, "source_md5": actual}


def tadi(root):
    report = {"source": "https://zenodo.org/records/8399829", "files": [], "loggers": []}
    for name, md5 in TADI.items():
        report["files"].append(checked(root / name, md5))
        if not name.startswith("Logger_"):
            continue
        with (root / name).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        release = collections.Counter(row["Release"] for row in rows)
        missing = {key: sum(row[key] in ("", "-9999", "-9999.0") for row in rows)
                   for key in rows[0] if key not in ("time", "Release")}
        report["loggers"].append({
            "file": name, "rows": len(rows), "first": rows[0]["time"],
            "last": rows[-1]["time"], "release_ids": sorted(release, key=int),
            "rows_with_blank_or_zero_release_id": sum(release.get(x, 0) for x in ("", "0")),
            "sentinel_or_blank_by_column": {k: v for k, v in missing.items() if v},
        })
    return report


def sb112(root):
    report = {"source": "https://zenodo.org/records/6616632", "files": [], "sheets": []}
    for name, md5 in SB112.items():
        report["files"].append(checked(root / name, md5))
        workbook = openpyxl.load_workbook(root / name, read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            iterator = iter(sheet.values)
            header = next(iterator)
            if header != ("time", "data_type", "unit", "data_value"):
                raise ValueError(f"Unexpected column names in {name}/{sheet.title}")
            rows = [row for row in iterator if row[0]]
            times = [dt.datetime.fromisoformat(row[0]) for row in rows]
            gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
            report["sheets"].append({
                "file": name, "sheet": sheet.title, "rows": len(rows),
                "first": rows[0][0], "last": rows[-1][0],
                "median_gap_s": sorted(gaps)[len(gaps) // 2] if gaps else None,
                "max_gap_s": max(gaps, default=None), "units": sorted({r[2] for r in rows}),
            })
        workbook.close()
    return report


if __name__ == "__main__":
    data = Path("data/raw")
    output = Path("results/stage7_e1/external_source_audit.json")
    report = {"tadi": tadi(data / "tadi_2019_zenodo_8399829"),
              "sb112": sb112(data / "sb112_zenodo_6616632")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", output, "TADI rows", sum(x["rows"] for x in report["tadi"]["loggers"]),
          "SB112 rows", sum(x["rows"] for x in report["sb112"]["sheets"]))
