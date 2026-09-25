"""Create A2's frozen frame manifest and a private human-annotation workspace.

Raw frames and derived images remain under data/processed/ (gitignored). The
manifest contains paths, timestamps, and group assignment, never person labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

SOURCES = (
    (
        "rgbd_bonn_person_tracking",
        "a4810fd91ef2ea1d630b53fe0df5d76144c1b18d86ca91fb3a035debd0c9c5f5",
        "development",
    ),
    (
        "rgbd_bonn_person_tracking2",
        "d3ef7898529c60dc39919ea699d00490d98a2c6ae4b165610f2955b235b939b5",
        "held_out_similar_room",
    ),
)
HEADERS = (
    "sample_id",
    "sequence",
    "frame_index",
    "split",
    "qc_double_label",
    "rgb_timestamp_s",
    "depth_timestamp_s",
    "pair_offset_ms",
    "rgb_zip_member",
    "depth_zip_member",
    "local_rgb_path",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read_rows(archive: ZipFile, member: str) -> list[tuple[float, str]]:
    return [
        (float(parts[0]), parts[1])
        for line in archive.read(member).decode("utf-8").splitlines()
        if (parts := line.strip().split()) and not line.lstrip().startswith("#")
    ]


def prepare(raw_dir: Path, output_dir: Path, manifest_path: Path) -> list[dict[str, object]]:
    # Run all eligibility checks before publishing a manifest or extracting frames.
    rows: list[dict[str, object]] = []
    images: list[tuple[Path, str, Path]] = []
    for sequence, expected_hash, split in SOURCES:
        zip_path = raw_dir / f"{sequence}.zip"
        if digest(zip_path) != expected_hash:
            raise ValueError(f"Original ZIP SHA-256 mismatch: {zip_path}")
        with ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            root = sequence + "/"
            rgb = read_rows(archive, root + "rgb.txt")
            depth = [pair for pair in read_rows(archive, root + "depth.txt") if root + pair[1] in names]
            if not rgb or not depth:
                raise ValueError(f"Missing RGB/depth records: {sequence}")
            used_depth: set[str] = set()
            for index in range(0, len(rgb), 10):
                rgb_time, rgb_relative = rgb[index]
                rgb_member = root + rgb_relative
                if rgb_member not in names:
                    raise ValueError(f"Missing selected RGB file: {rgb_member}")
                depth_time, depth_relative = min(depth, key=lambda item: abs(item[0] - rgb_time))
                depth_member = root + depth_relative
                offset_ms = abs(rgb_time - depth_time) * 1000
                if offset_ms > 17.0 or depth_member in used_depth:
                    raise ValueError(f"Depth alignment failed for {sequence} frame {index}")
                used_depth.add(depth_member)
                local_image = f"frames/{sequence}/{index:04d}.png"
                rows.append(
                    {
                        "sample_id": f"{sequence}:{index:04d}",
                        "sequence": sequence,
                        "frame_index": index,
                        "split": split,
                        "qc_double_label": int(index % 50 == 0),
                        "rgb_timestamp_s": f"{rgb_time:.5f}",
                        "depth_timestamp_s": f"{depth_time:.5f}",
                        "pair_offset_ms": f"{offset_ms:.3f}",
                        "rgb_zip_member": rgb_member,
                        "depth_zip_member": depth_member,
                        "local_rgb_path": local_image,
                    }
                )
                images.append((zip_path, rgb_member, output_dir / local_image))

    if len(rows) != 115 or sum(int(row["qc_double_label"]) for row in rows) != 24:
        raise ValueError("The approved sample/second-annotator counts changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(Path(__file__).resolve().parent.parent / "tools/annotate_bonn_v1a.html", output_dir / "annotate.html")

    # Each archive is opened once, avoiding repeated decompression of huge ZIPs.
    for sequence, _, _ in SOURCES:
        zip_path = raw_dir / f"{sequence}.zip"
        with ZipFile(zip_path) as archive:
            for source, member, destination in images:
                if source != zip_path:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/bonn_rgbd"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/bonn_v1a"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/audits/bonn_v1a_sampling_manifest.csv")
    )
    args = parser.parse_args()
    rows = prepare(args.raw_dir, args.output_dir, args.manifest)
    print(f"Selected {len(rows)} RGB frames, including {sum(int(r['qc_double_label']) for r in rows)} independent review frames")
    print(f"Manifest: {args.manifest}; private annotation workspace: {args.output_dir}")


if __name__ == "__main__":
    main()
