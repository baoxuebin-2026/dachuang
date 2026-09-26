#!/usr/bin/env python3
"""Download verified UCI archives without third-party Python packages."""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "309": (
        "gas+sensor+array+exposed+to+turbulent+gas+mixtures",
        "5e9b707e7a44b3dcaf39b62bb46597403fcdd98ec89814b15e4876694db5f41e",
    ),
    "322": (
        "gas+sensor+array+under+dynamic+gas+mixtures",
        "8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7",
    ),
    "362": (
        "gas+sensors+for+home+activity+monitoring",
        "7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395",
    ),
    "487": (
        "gas+sensor+array+temperature+modulation",
        "cb5ce4a6af1a51b933d1979952d7845f0e9baac54e15a81a1e3599e8b85905d4",
    ),
}

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset_id: str) -> None:
    slug, expected = DATASETS[dataset_id]
    url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/{slug}.zip"
    DEST.mkdir(parents=True, exist_ok=True)
    target = (DEST / "uci362" / "uci362_original.zip" if dataset_id == "362"
              else DEST / f"uci_{dataset_id}_original.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".zip.part")

    if target.exists() and sha256(target) == expected:
        print(f"{dataset_id}: already verified: {target}")
        return

    print(f"{dataset_id}: downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)

        actual = sha256(partial)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: expected {expected}, received {actual}")

        with zipfile.ZipFile(partial) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"ZIP checksum failed at {corrupt}")
        partial.replace(target)
        print(f"{dataset_id}: verified SHA-256 and ZIP CRC: {target}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=(*DATASETS, "all"), help="UCI dataset ID")
    args = parser.parse_args()
    for dataset_id in DATASETS if args.dataset == "all" else [args.dataset]:
        download(dataset_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
