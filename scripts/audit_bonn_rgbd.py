"""Audit an official Bonn RGB-D Dynamic ZIP before visual experiment design.

Only inspects provenance-relevant file structure and timestamp alignment. It does
not infer person boxes, people positions, or camera accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
from pathlib import Path
from zipfile import ZipFile


def read_manifest(archive: ZipFile, prefix: str, name: str) -> list[tuple[float, str]]:
    lines = archive.read(prefix + name).decode("utf-8").splitlines()
    return [
        (float(parts[0]), parts[1])
        for raw in lines
        if (parts := raw.strip().split()) and not raw.lstrip().startswith("#")
    ]


def png_header(archive: ZipFile, name: str) -> dict[str, int]:
    with archive.open(name) as stream:
        header = stream.read(26)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG: {name}")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "png_color_type": color_type,
    }


def audit(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)

    with ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC failed: {corrupt}")
        roots = {name.split("/", 1)[0] for name in archive.namelist()}
        if len(roots) != 1:
            raise ValueError(f"Expected one sequence root, found: {sorted(roots)}")
        root = roots.pop() + "/"
        names = set(archive.namelist())
        rgb = read_manifest(archive, root, "rgb.txt")
        depth = read_manifest(archive, root, "depth.txt")
        poses = [
            line
            for line in archive.read(root + "groundtruth.txt").decode("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        present_rgb = [(time, root + rel) for time, rel in rgb if root + rel in names]
        present_depth = [(time, root + rel) for time, rel in depth if root + rel in names]
        missing_rgb = [rel for _, rel in rgb if root + rel not in names]
        missing_depth = [rel for _, rel in depth if root + rel not in names]
        if not present_rgb or not present_depth:
            raise ValueError("No RGB/depth pairs can be checked")
        sample_rgb = png_header(archive, present_rgb[0][1])
        sample_depth = png_header(archive, present_depth[0][1])
        matches = [
            min(range(len(present_depth)), key=lambda j: abs(present_depth[j][0] - time))
            for time, _ in present_rgb
        ]
        errors_ms = [
            abs(time - present_depth[index][0]) * 1000
            for (time, _), index in zip(present_rgb, matches, strict=True)
        ]

    return {
        "source_file": path.name,
        "archive_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "zip_crc_ok": True,
        "sequence": root.rstrip("/"),
        "rgb_manifest_rows": len(rgb),
        "depth_manifest_rows": len(depth),
        "camera_pose_rows": len(poses),
        "rgb_files": len(present_rgb),
        "depth_files": len(present_depth),
        "missing_rgb_manifest_paths": missing_rgb,
        "missing_depth_manifest_paths": missing_depth,
        "sample_rgb_png": sample_rgb,
        "sample_depth_png": sample_depth,
        "rgb_duration_s": round(present_rgb[-1][0] - present_rgb[0][0], 5),
        "nearest_depth_unique_matches": len(set(matches)),
        "nearest_depth_offset_median_ms": round(statistics.median(errors_ms), 3),
        "nearest_depth_offset_max_ms": round(max(errors_ms), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Official Bonn sequence ZIP")
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
