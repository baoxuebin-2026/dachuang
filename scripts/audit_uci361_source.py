#!/usr/bin/env python3
"""Read-only time/filename audit of UCI 361; never inspect sensor responses."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/uci361/uci361_original.zip"
OUTPUT = ROOT / "results/stage9_uci361/source_audit.json"
NAME = re.compile(r"^data1/B(?P<board>[1-5])_G(?P<gas>CO|Me|Ea|Ey)_F(?P<level>\d{3})_R(?P<repeat>[1-4])\.txt$")
EXPECTED_SHA = "e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167"


def main() -> None:
    sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError("Unexpected UCI 361 original ZIP")
    records = []
    with zipfile.ZipFile(ARCHIVE) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        names = sorted(n for n in z.namelist() if n.endswith(".txt"))
        if len(names) != 640:
            raise ValueError(f"Expected 640 recordings, found {len(names)}")
        for name in names:
            m = NAME.fullmatch(name)
            if m is None:
                raise ValueError(name)
            count = 0
            prev = None
            start = end = None
            reversed_time = duplicates = 0
            with z.open(name) as f:
                for line in f:
                    words = line.split()
                    if len(words) != 9:
                        raise ValueError(f"Unexpected number of columns: {name}")
                    time = float(words[0])
                    if count == 0:
                        start = time
                    if prev is not None:
                        reversed_time += int(time < prev)
                        duplicates += int(time == prev)
                    end = prev = time
                    count += 1
            records.append({"name": name, **m.groupdict(), "rows": count,
                            "first_time_s": start, "last_time_s": end,
                            "backward_time_rows": reversed_time,
                            "duplicate_time_rows": duplicates})
    result = {
        "source": "UCI 361 original ZIP; source-only time and filename audit",
        "sha256": sha, "zip_size_bytes": ARCHIVE.stat().st_size,
        "recordings": len(records), "by_gas": dict(Counter(r["gas"] for r in records)),
        "by_board": dict(Counter(r["board"] for r in records)),
        "board_repeat_groups": len({(r["board"], r["repeat"]) for r in records}),
        "rows_range": [min(r["rows"] for r in records), max(r["rows"] for r in records)],
        "backward_time_rows": sum(r["backward_time_rows"] for r in records),
        "duplicate_time_rows": sum(r["duplicate_time_rows"] for r in records),
        "duration_range_s": [min(r["last_time_s"] for r in records),
                             max(r["last_time_s"] for r in records)],
        "records": records,
        "note": "The fixed 50s clean-air / 100s stimulus / 450s purge schedule comes from the source paper, not the text files. No sensor scores or detections were calculated.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
