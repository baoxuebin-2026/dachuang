import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.request

RECORD = int(sys.argv[1]) if len(sys.argv)>1 else 4264224
NAME = sys.argv[2] if len(sys.argv)>2 else "dataset.mat"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{NAME}/content"
META = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}", timeout=30))
FILE = next(f for f in META["files"] if f["key"] == NAME)
SIZE = FILE["size"]
PARTS = 16 if SIZE>1000000000 else 8
STEP = (SIZE + PARTS - 1) // PARTS
STEM = f"zenodo_{RECORD}_{NAME}"


def get(i):
    start = i * STEP
    end = min((i + 1) * STEP, SIZE) - 1
    path = f"{STEM}.part{i:02}"
    if os.path.exists(path) and os.stat(path).st_size == end - start + 1:
        return path
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as response, open(path + ".tmp", "wb") as out:
                assert response.status == 206
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            assert os.stat(path + ".tmp").st_size == end - start + 1
            os.replace(path + ".tmp", path)
            print(f"part {i} done", flush=True)
            return path
        except Exception as exc:
            print(f"part {i} attempt {attempt + 1}: {exc}", flush=True)
    raise RuntimeError(f"Could not fetch part {i}")


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(get, range(PARTS)))

hasher = hashlib.md5()
with open(f"{STEM}", "wb") as out:
    for i in range(PARTS):
        with open(f"{STEM}.part{i:02}", "rb") as source:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
                hasher.update(chunk)
print("size", os.stat(STEM).st_size, "md5", hasher.hexdigest(), "expected", FILE["checksum"], flush=True)
assert FILE["checksum"].split(":")[-1] == hasher.hexdigest()
