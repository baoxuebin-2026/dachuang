#!/usr/bin/env python3
"""Read-only source and independence audit of the UCI 362 original ZIP.

No response scores, fitted thresholds, model outputs, or performance metrics.
The source archive stays in data/raw; the small JSON summary may be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_metadata(payload: bytes) -> dict[int, dict]:
    lines = payload.decode("utf-8").splitlines()
    if lines[0].split() != ["id", "date", "class", "t0", "dt"]:
        raise ValueError("Unexpected metadata header")
    records: dict[int, dict] = {}
    for line in lines[1:]:
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid metadata row: {line!r}")
        ident = int(values[0])
        if ident in records:
            raise ValueError(f"Duplicate metadata ID {ident}")
        day = datetime.strptime(values[1], "%m-%d-%y")
        records[ident] = {"date": values[1], "day": day,
                          "class": values[2], "t0_hours": float(values[3]),
                          "dt_hours": float(values[4])}
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path("data/raw/uci362/uci362_original.zip"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/stage9_uci362/source_audit.json"))
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as outer:
        if outer.testzip() is not None:
            raise ValueError("Corrupt outer ZIP")
        metadata = read_metadata(outer.read("HT_Sensor_metadata.dat"))
        with zipfile.ZipFile(io.BytesIO(outer.read("HT_Sensor_dataset.zip"))) as inner:
            if inner.testzip() is not None:
                raise ValueError("Corrupt inner ZIP")
            counts: Counter[int] = Counter()
            stats: dict[int, dict] = defaultdict(lambda: {
                "time_min_hours": None, "time_max_hours": None,
                "nonmonotone_rows": 0, "duplicate_time_rows": 0,
                "gap_over_2_5s_rows": 0, "max_gap_s": 0.0,
                "nonfinite_sensor_rows": 0, "nonpositive_sensor_rows": 0,
                "nonfinite_ambient_rows": 0,
            })
            last_time: dict[int, float] = {}
            second_bins: dict[int, set[int]] = defaultdict(set)
            seen_last_id: int | None = None
            interleaved_id_rows = 0
            with inner.open("HT_Sensor_dataset.dat") as stream:
                header = stream.readline().decode("utf-8").split()
                if header != ["id", "time", *(f"R{i}" for i in range(1, 9)),
                              "Temp.", "Humidity"]:
                    raise ValueError(f"Unexpected time-series header: {header}")
                for line_number, line in enumerate(stream, start=2):
                    words = line.split()
                    if len(words) != 12:
                        raise ValueError(f"Invalid field count at {line_number}")
                    ident = int(words[0])
                    if ident not in metadata:
                        raise ValueError(f"No metadata for ID {ident}")
                    values = [float(word) for word in words[1:]]
                    time, sensors, ambient = values[0], values[1:9], values[9:]
                    if not math.isfinite(time):
                        raise ValueError(f"Nonfinite time at {line_number}")
                    record = stats[ident]
                    counts[ident] += 1
                    second_bins[ident].add(math.floor(time * 3600))
                    if seen_last_id is not None and ident < seen_last_id:
                        interleaved_id_rows += 1
                    seen_last_id = ident
                    if ident in last_time:
                        record["nonmonotone_rows"] += int(time < last_time[ident])
                        record["duplicate_time_rows"] += int(time == last_time[ident])
                        gap_s = (time - last_time[ident]) * 3600
                        record["gap_over_2_5s_rows"] += int(gap_s > 2.5)
                        record["max_gap_s"] = max(record["max_gap_s"], gap_s)
                    last_time[ident] = time
                    if record["time_min_hours"] is None:
                        record["time_min_hours"] = time
                    record["time_max_hours"] = time
                    record["nonfinite_sensor_rows"] += int(not all(math.isfinite(x) for x in sensors))
                    record["nonpositive_sensor_rows"] += int(any(math.isfinite(x) and x <= 0 for x in sensors))
                    record["nonfinite_ambient_rows"] += int(not all(math.isfinite(x) for x in ambient))

    trials: dict[int, dict] = {}
    for ident in sorted(counts):
        meta = metadata[ident]
        rec = stats[ident]
        bins = second_bins[ident]
        # t0 is recorded to 0.01 h (~36 s), so interval overlap is a
        # conservative screen, never proof of identical raw samples.
        start = meta["day"] + timedelta(hours=meta["t0_hours"] + rec["time_min_hours"])
        end = meta["day"] + timedelta(hours=meta["t0_hours"] + rec["time_max_hours"])
        trials[ident] = {
            "date": meta["date"], "class": meta["class"],
            "t0_hours": meta["t0_hours"], "dt_hours": meta["dt_hours"],
            "rows": counts[ident], **rec,
            "calibration_bins_15min": sum(-3600 <= s < -2700 for s in bins),
            "train_background_bins_40min": sum(-2700 <= s < -300 for s in bins),
            "pre_background_bins_45min": sum(-2700 <= s < 0 for s in bins),
            "post_stimulus_bins_5min": sum(0 <= s < 300 for s in bins),
            "estimated_start": start.isoformat(), "estimated_end": end.isoformat(),
        }

    # Connect all overlapping recording spans; also connect equal metadata
    # dates even when sampled intervals do not overlap.
    ids = list(trials)
    parent = {i: i for i in ids}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(j)] = find(i)

    same_date_pairs = overlapping_pairs = 0
    for at, i in enumerate(ids):
        a = trials[i]
        for j in ids[at + 1:]:
            b = trials[j]
            same_date = a["date"] == b["date"]
            overlap = a["estimated_start"] <= b["estimated_end"] and b["estimated_start"] <= a["estimated_end"]
            same_date_pairs += int(same_date)
            overlapping_pairs += int(overlap)
            if same_date or overlap:
                union(i, j)

    components: dict[int, list[int]] = defaultdict(list)
    for ident in ids:
        components[find(ident)].append(ident)
    clusters = sorted((sorted(members) for members in components.values()), key=lambda x: x[0])
    output = {
        "source": "UCI 362 original ZIP, official UCI repository",
        "sha256": digest(args.archive),
        "outer_size_bytes": args.archive.stat().st_size,
        "metadata_count": len(metadata), "timeseries_id_count": len(ids),
        "metadata_only_ids": sorted(set(metadata) - set(ids)),
        "metadata_class_counts": dict(sorted(Counter(x["class"] for x in metadata.values()).items())),
        "paired_class_counts": dict(sorted(Counter(x["class"] for x in trials.values()).items())),
        "raw_rows": sum(counts.values()),
        "unique_metadata_dates": len({m["date"] for m in metadata.values()}),
        "unique_paired_dates": len({m["date"] for m in trials.values()}),
        "same_date_pairs": same_date_pairs,
        "estimated_overlap_pairs": overlapping_pairs,
        "connected_group_count": len(clusters),
        "connected_groups": clusters,
        "interleaved_id_rows": interleaved_id_rows,
        "nonmonotone_rows": sum(x["nonmonotone_rows"] for x in trials.values()),
        "duplicate_time_rows": sum(x["duplicate_time_rows"] for x in trials.values()),
        "gap_over_2_5s_rows": sum(x["gap_over_2_5s_rows"] for x in trials.values()),
        "max_gap_s": max(x["max_gap_s"] for x in trials.values()),
        "nonfinite_sensor_rows": sum(x["nonfinite_sensor_rows"] for x in trials.values()),
        "nonpositive_sensor_rows": sum(x["nonpositive_sensor_rows"] for x in trials.values()),
        "nonfinite_ambient_rows": sum(x["nonfinite_ambient_rows"] for x in trials.values()),
        "trials": {str(k): v for k, v in trials.items()},
        "note": "t0 has 0.01-hour precision; estimated overlaps are a grouping screen, not exact identity. No model was run.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in ("trials", "connected_groups")}, indent=2))


if __name__ == "__main__":
    main()
