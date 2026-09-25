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
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
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
DOWNLOADS = {
    "rgbd_bonn_person_tracking": (
        "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking.zip",
        329_482_910,
    ),
    "rgbd_bonn_person_tracking2": (
        "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking2.zip",
        324_262_783,
    ),
}
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


def ensure_archive(path: Path, expected_hash: str, url: str, size: int, *, offline: bool) -> None:
    if path.exists():
        if path.stat().st_size != size or digest(path) != expected_hash:
            raise ValueError(
                f"已有文件大小或 SHA-256 错误：{path}。请核对下载来源；不会自动覆盖它。"
            )
        return
    if offline:
        raise FileNotFoundError(f"缺少原始文件：{path}；官网地址：{url}")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() and partial.stat().st_size > size:
        raise ValueError(f"断点文件超过官方长度：{partial}")
    print(f"缺少 {path.name}，从官方来源下载 {size:,} 字节；中断后重新运行可续传。", flush=True)
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        request = Request(url, headers={"Range": f"bytes={offset}-", "User-Agent": "Bonn-V1A-research/1.0"})
        try:
            with urlopen(request, timeout=45) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{size}"):
                        raise ValueError(f"服务器返回了不一致的 Range 响应：{content_range}")
                    mode = "ab"
                elif response.status == 200:
                    # Some proxies ignore Range. Restart rather than concatenate two files.
                    mode = "wb"
                    offset = 0
                else:
                    raise ValueError(f"无法确认下载响应：HTTP {response.status}")
                with partial.open(mode) as output:
                    next_notice = ((offset // (32 * 1024 * 1024)) + 1) * (32 * 1024 * 1024)
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        offset += len(chunk)
                        if offset > size:
                            raise ValueError(f"下载长度超过官方长度：{partial}")
                        if offset >= next_notice:
                            print(f"  {path.name}: {offset:,}/{size:,} 字节", flush=True)
                            next_notice += 32 * 1024 * 1024
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            print(f"  下载中断（第 {attempt}/8 次）：{error}", flush=True)
            if attempt < 8:
                time.sleep(min(attempt, 4))
            continue
        if partial.stat().st_size == size:
            break
    if not partial.exists() or partial.stat().st_size != size:
        raise RuntimeError(f"文件尚未完整：{partial}。请重试，或手动从 {url} 下载后放到 {path}")
    if digest(partial) != expected_hash:
        raise ValueError(f"文件 SHA-256 不符：{partial}。保留断点文件供检查，不会使用此文件。")
    partial.replace(path)
    print(f"  完成并通过 SHA-256 校验：{path}", flush=True)


def read_rows(archive: ZipFile, member: str) -> list[tuple[float, str]]:
    return [
        (float(parts[0]), parts[1])
        for line in archive.read(member).decode("utf-8").splitlines()
        if (parts := line.strip().split()) and not line.lstrip().startswith("#")
    ]


def prepare(
    raw_dir: Path, output_dir: Path, manifest_path: Path, *, offline: bool = False
) -> list[dict[str, object]]:
    # Run all eligibility checks before publishing a manifest or extracting frames.
    rows: list[dict[str, object]] = []
    images: list[tuple[Path, str, Path]] = []
    for sequence, expected_hash, split in SOURCES:
        zip_path = raw_dir / f"{sequence}.zip"
        url, size = DOWNLOADS[sequence]
        ensure_archive(zip_path, expected_hash, url, size, offline=offline)
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
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data/raw/bonn_rgbd")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data/processed/bonn_v1a")
    parser.add_argument(
        "--manifest", type=Path, default=repo_root / "data/audits/bonn_v1a_sampling_manifest.csv"
    )
    parser.add_argument("--offline", action="store_true", help="只用本地文件，不自动下载缺失的官方 ZIP")
    args = parser.parse_args()
    rows = prepare(args.raw_dir, args.output_dir, args.manifest, offline=args.offline)
    print(f"Selected {len(rows)} RGB frames, including {sum(int(r['qc_double_label']) for r in rows)} independent review frames")
    print(f"Manifest: {args.manifest}; private annotation workspace: {args.output_dir}")


if __name__ == "__main__":
    main()
