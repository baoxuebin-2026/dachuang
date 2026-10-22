"""Frozen protocol, before inspecting any board response values or model outputs.

Independent 2024 Chongqing University E-nose data: 37 features, 14 odor
labels, two boards (before/after). It does NOT publish a gas-name mapping or
sample concentration in this archive. This is a cross-board mechanism stress
test only; it cannot substantiate low-concentration CO recognition.

Evaluation unit: complete labeled odor exposure, never feature/channel rows.
In each of two directions, train on one entire board and test on the other.
Use all 14 classes, no target-board fitting or tuning. For both directions
compare same C=1 RBF SVM on signed-log raw features, row-normalized features,
and the candidate raw-feature amplitude augmentation at 0.5, 0.75, 1.0.
StandardScaler is fitted to source only. Archive MD5 is mandatory. Fixed
criterion: augmentation wins only if macro F1 improves by >=0.05 over the
stronger baseline in both directions and class recalls lose <=0.05 each.
Do not call repeated 15 exposures independent boards or infer concentrations.
"""

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "chongqing_board_source.rar"
EXPECTED_MD5 = "bba2d8537aef4617acfc3598a6bb86db"


def extract_board_mats(path):
    lib = ctypes.CDLL("libarchive.so.13")
    bindings = {
        "archive_read_new": ([], ctypes.c_void_p),
        "archive_read_support_filter_all": ([ctypes.c_void_p], ctypes.c_int),
        "archive_read_support_format_all": ([ctypes.c_void_p], ctypes.c_int),
        "archive_read_open_filename": ([ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int),
        "archive_read_next_header": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
        "archive_entry_pathname": ([ctypes.c_void_p], ctypes.c_char_p),
        "archive_entry_size": ([ctypes.c_void_p], ctypes.c_int64),
        "archive_read_data": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_ssize_t),
        "archive_read_free": ([ctypes.c_void_p], ctypes.c_int),
    }
    for name, (args, result) in bindings.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    a = lib.archive_read_new()
    try:
        lib.archive_read_support_filter_all(a)
        lib.archive_read_support_format_all(a)
        if lib.archive_read_open_filename(a, str(path).encode(), 10240) != 0:
            raise ValueError("Could not open RAR archive")
        output = {}
        entry = ctypes.c_void_p()
        while lib.archive_read_next_header(a, ctypes.byref(entry)) == 0:
            name = lib.archive_entry_pathname(entry).decode()
            if name not in ("board drift/data_lab_before.mat", "board drift/data_lab_after.mat"):
                continue
            size = lib.archive_entry_size(entry)
            buffer = ctypes.create_string_buffer(size)
            if lib.archive_read_data(a, buffer, size) != size:
                raise ValueError(f"Incomplete archive member: {name}")
            import io
            m = loadmat(io.BytesIO(buffer.raw))
            if "before" in name:
                output["before"] = (m["data"], m["numlab"].ravel())
            else:
                output["after"] = (m["dataafterdrift"], m["numlabafterdrift"].ravel())
        return output
    finally:
        lib.archive_read_free(a)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def main():
    md5 = hashlib.md5(ARCHIVE.read_bytes()).hexdigest()
    if md5 != EXPECTED_MD5:
        raise ValueError(f"Unexpected archive MD5 {md5}")
    boards = extract_board_mats(ARCHIVE)
    assert set(boards) == {"before", "after"}
    results = {}
    for source_name, target_name in (("before", "after"), ("after", "before")):
        raw, y = boards[source_name]
        test_raw, test_y = boards[target_name]
        if raw.shape[1] != 37 or test_raw.shape[1] != 37 or set(y) != set(range(1, 15)) or set(test_y) != set(range(1, 15)):
            raise ValueError("Unexpected class labels or number of sensor features")
        specs = {
            "raw_svm": (signed_log(raw), signed_log(test_raw), False, y),
            "normalized_svm": (signed_log(raw), signed_log(test_raw), True, y),
            "amplitude_augmented_svm": (signed_log(np.concatenate([raw * a for a in (0.5, 0.75, 1.0)])), signed_log(test_raw), False, np.tile(y, 3)),
        }
        fold = {"n_source": len(y), "n_target": len(test_y), "models": {}}
        for name, (train, test, norm, labels) in specs.items():
            steps = ([Normalizer()] if norm else []) + [StandardScaler(), SVC(C=1, gamma="scale")]
            model = make_pipeline(*steps).fit(train, labels)
            pred = model.predict(test)
            fold["models"][name] = {
                "macro_f1": float(f1_score(test_y, pred, labels=range(1, 15), average="macro", zero_division=0)),
                "class_recall": recall_score(test_y, pred, labels=range(1, 15), average=None, zero_division=0).tolist(),
                "confusion": confusion_matrix(test_y, pred, labels=range(1, 15)).tolist(),
            }
        baseline = max((fold["models"][n] for n in ("raw_svm", "normalized_svm")), key=lambda v: v["macro_f1"])
        ours = fold["models"]["amplitude_augmented_svm"]
        fold["passes"] = ours["macro_f1"] >= baseline["macro_f1"] + 0.05 and all(a >= b - 0.05 for a, b in zip(ours["class_recall"], baseline["class_recall"]))
        results[source_name + "_to_" + target_name] = fold
        print(source_name, "->", target_name, {k: round(v["macro_f1"], 4) for k, v in fold["models"].items()}, "passes", fold["passes"])
    output = {"protocol": __doc__, "source": "https://doi.org/10.5281/zenodo.13854443", "md5": md5, "passes": all(f["passes"] for f in results.values()), "results": results}
    (ROOT / "chongqing_board_amplitude_stress_2026_10_22.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print("Overall passes", output["passes"])


if __name__ == "__main__":
    main()
