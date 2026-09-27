"""Download the original TADI 2019 files from Zenodo and verify MD5 checksums."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


RECORD = "https://zenodo.org/records/8399829/files"
FILES = {
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


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/tadi_2019"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, expected in FILES.items():
        target = args.output_dir / name
        if target.exists() and md5(target) == expected:
            print(f"verified {target}")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with urlopen(f"{RECORD}/{name}?download=1", timeout=60) as response:
                with temporary.open("wb") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            observed = md5(temporary)
            if observed != expected:
                raise ValueError(f"MD5 mismatch for {name}: {observed} != {expected}")
            temporary.replace(target)
            print(f"downloaded {target}")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
