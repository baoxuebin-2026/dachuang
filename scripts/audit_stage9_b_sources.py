#!/usr/bin/env python3
"""Source-only audit of two untouched Mendeley candidates; no model is trained.

Requires the original archives under data/raw/mendeley_<dataset_id>/.
The RAR branch uses the system libarchive for inspection, without extraction.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import re
import zipfile
from collections import Counter
from ctypes.util import find_library
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage9_b_sources"
MINE_ID = "yd7vw4c5mk"
PIPE_ID = "sdh7x7b68d"
MINE_SHA = "405a8b27004a2e318bf213cf209398d3d58b45bb3c7a78f159e69003ff459311"
PIPE_SHA = "d1a863309156033ed5f5ce909f464078cca6ab247b775e66254c0db8b77dffc5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(name: str, content: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    if target.exists():
        raise FileExistsError(f"Existing audit will not be overwritten: {target}")
    content["audit_code_sha256"] = sha256(Path(__file__))
    target.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in content.items() if k not in ("per_day", "files")}, indent=2))


def mine() -> None:
    archive = ROOT / f"data/raw/mendeley_{MINE_ID}/methane_data.zip"
    attr = ROOT / f"data/raw/mendeley_{MINE_ID}/attribute_information.txt"
    if sha256(archive) != MINE_SHA:
        raise ValueError("Methane ZIP differs from publisher SHA-256")
    if sha256(attr) != "0eed822b03816dee2e437fc3fde7eb12a25b6b236c3d107cedddcb747404ee05":
        raise ValueError("Attribute text differs from publisher SHA-256")
    counts = Counter()
    target_names = ("MM263", "MM264", "MM256")
    warning_rows = Counter()
    warning_starts = Counter()
    long_warning_runs = Counter()
    run_lengths = Counter()
    previous_warning = {x: False for x in target_names}
    previous_stamp = None
    first = last = None
    last_day = None
    cached_day_number = None
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != ["methane_data.csv"] or z.testzip() is not None:
            raise ValueError("Unexpected ZIP contents or CRC failure")
        with z.open("methane_data.csv") as stream:
            lines = (line.decode("utf-8-sig") for line in stream)
            reader = csv.reader(lines)
            header = next(reader)
            if header[:6] != ["year", "month", "day", "hour", "minute", "second"] or len(header) != 34:
                raise ValueError("Unexpected 34-column mine schema")
            indices = {name: header.index(name) for name in target_names}
            for row in reader:
                if len(row) != len(header):
                    counts["wrong_width"] += 1
                    continue
                day_key = tuple(int(x) for x in row[:3])
                if day_key != last_day:
                    cached_day_number = date(*day_key).toordinal()
                    last_day = day_key
                    counts["days"] += 1
                stamp = cached_day_number * 86400 + int(row[3]) * 3600 + int(row[4]) * 60 + int(row[5])
                if first is None:
                    first = day_key + tuple(int(x) for x in row[3:6])
                last = day_key + tuple(int(x) for x in row[3:6])
                gap = previous_stamp is not None and stamp != previous_stamp + 1
                if previous_stamp is not None:
                    counts["duplicate_or_backward"] += int(stamp <= previous_stamp)
                    counts["forward_gaps"] += int(stamp > previous_stamp + 1)
                previous_stamp = stamp
                counts["rows"] += 1
                for name, ix in indices.items():
                    value = float(row[ix])
                    if not 0 <= value <= 30:
                        counts[f"{name}_outside_0_30"] += 1
                    # Only source-label feasibility: measured warning crossing,
                    # never a ground-truth release or a classifier score.
                    above = value >= 1.0
                    warning_rows[name] += int(above)
                    if gap and previous_warning[name] and run_lengths[name] >= 30:
                        long_warning_runs[name] += 1
                    if above:
                        run_lengths[name] = 1 if gap or not previous_warning[name] else run_lengths[name] + 1
                        if gap or not previous_warning[name]:
                            warning_starts[name] += 1
                    elif previous_warning[name] and run_lengths[name] >= 30:
                        long_warning_runs[name] += 1
                    previous_warning[name] = above
                if gap:
                    counts["timestamp_discontinuity_breaks_event_runs"] += 1
            for name in target_names:
                if previous_warning[name] and run_lengths[name] >= 30:
                    long_warning_runs[name] += 1
    save("mine_source_audit.json", {
        "dataset": MINE_ID, "source": "Mendeley Data DOI 10.17632/yd7vw4c5mk.1",
        "archive_bytes": archive.stat().st_size, "archive_sha256": MINE_SHA,
        "attribute_sha256": sha256(attr), "columns": header, "first_timestamp": first,
        "last_timestamp": last, "counts": dict(counts),
        "warning_threshold_pct_ch4": 1.0, "warning_rows_from_measured_target": dict(warning_rows),
        "warning_episode_starts": dict(warning_starts),
        "warning_episode_runs_at_least_30_observed_seconds": dict(long_warning_runs),
        "note": "Publisher-synchronised 1s time series. Crossings are defined from the same measured methane sensors, not independent release/accident truth. No train/test division, model, or risk score calculated.",
    })


def libarchive_entries(path: Path):
    libname = find_library("archive")
    if libname is None:
        raise RuntimeError("libarchive is required to inspect RAR v5; source archive remains intact")
    lib = ctypes.CDLL(libname)
    lib.archive_read_new.restype = ctypes.c_void_p
    for name in ("archive_read_support_filter_all", "archive_read_support_format_all", "archive_read_free"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.archive_read_open_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.archive_entry_pathname.argtypes = [ctypes.c_void_p]
    lib.archive_entry_pathname.restype = ctypes.c_char_p
    lib.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.archive_read_data.restype = ctypes.c_ssize_t
    archive = lib.archive_read_new()
    try:
        lib.archive_read_support_filter_all(archive)
        lib.archive_read_support_format_all(archive)
        if lib.archive_read_open_filename(archive, str(path).encode(), 10240) != 0:
            raise ValueError("Unable to open RAR archive")
        entry = ctypes.c_void_p()
        while True:
            status = lib.archive_read_next_header(archive, ctypes.byref(entry))
            if status == 1:
                break
            if status != 0:
                raise ValueError(f"RAR header error {status}")
            name = lib.archive_entry_pathname(entry).decode(errors="replace")
            if not name.endswith(".txt"):
                continue
            chunks = []
            buffer = ctypes.create_string_buffer(1 << 16)
            while True:
                n = lib.archive_read_data(archive, buffer, len(buffer))
                if n < 0:
                    raise ValueError(f"RAR payload read error in {name}")
                if n == 0:
                    break
                chunks.append(buffer.raw[:n])
            yield name, b"".join(chunks)
    finally:
        lib.archive_read_free(archive)


def pipeline() -> None:
    paths = list((ROOT / f"data/raw/mendeley_{PIPE_ID}").glob("*.rar"))
    if len(paths) != 1 or sha256(paths[0]) != PIPE_SHA:
        raise ValueError("Pipeline RAR differs from publisher SHA-256")
    files = []
    levels = [set(), set(), set()]
    row_lengths = Counter()
    blank_lines = 0
    basename = re.compile(r"^(\d+)-(\d+)-(\d+)\.txt$")
    for name, raw in libarchive_entries(paths[0]):
        if not name.endswith(".txt"):
            continue
        m = basename.fullmatch(name.rsplit("/", 1)[-1])
        if m is None:
            raise ValueError(f"Unexpected RAR member: {name}")
        label = tuple(map(int, m.groups()))
        for i, level in enumerate(label):
            levels[i].add(level)
        lines = raw.decode("utf-8").splitlines()
        valid = [line for line in lines if line.strip()]
        if any(len(line.split("\t")) != 3 for line in valid):
            raise ValueError(f"Unexpected trial width in {name}")
        blank_lines += len(lines) - len(valid)
        row_lengths[len(valid)] += 1
        files.append({"name": name, "filename_concentrations": label,
                      "valid_rows": len(valid), "blank_rows": len(lines) - len(valid),
                      "columns": 3})
    if len(files) != 186:
        raise ValueError(f"Expected 186 trials, received {len(files)}")
    save("pipeline_source_audit.json", {
        "dataset": PIPE_ID, "source": "Mendeley Data DOI 10.17632/sdh7x7b68d.1",
        "archive_bytes": paths[0].stat().st_size, "archive_sha256": PIPE_SHA,
        "trial_files": len(files), "valid_rows_per_trial": dict(row_lengths),
        "trailing_blank_rows_total": blank_lines,
        "filename_levels_in_order": [sorted(x) for x in levels],
        "all_zero_concentration_trials": sum(x["filename_concentrations"] == (0, 0, 0) for x in files),
        "no_explicit_timestamp_column": True,
        "files": files,
        "note": "Three data columns are not labelled in raw TXT; file names encode three nominal concentrations. 50s clean / 200s injection / 250s purge timing comes from Mendeley description, not per-row event annotation. No sensor response or model score calculated.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("mine", "pipeline"))
    choice = parser.parse_args().source
    mine() if choice == "mine" else pipeline()
