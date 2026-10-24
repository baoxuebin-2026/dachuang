#!/usr/bin/env python3
"""Download and verify the three public datasets used by the gas paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "public_datasets_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for item in manifest["datasets"]:
        destination = args.output / item["filename"]
        if destination.exists() and not args.force:
            if destination.stat().st_size == item["bytes"] and sha256(destination) == item["sha256"]:
                print(f"verified {destination}")
                continue
            raise ValueError(f"Existing file failed verification: {destination}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"downloading UCI {item['uci_id']} -> {destination}", flush=True)
        urllib.request.urlretrieve(item["url"], temporary)
        if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Downloaded file failed verification: UCI {item['uci_id']}")
        temporary.replace(destination)
        print(f"verified {destination}")


if __name__ == "__main__":
    main()
