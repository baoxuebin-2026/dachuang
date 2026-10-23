"""Frozen UCI309 CO/CH4 low-flow mixture stress for the UCI361 manuscript."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "uci309_external_stress_protocol_2026_10_23.json").read_text())
ARCHIVE = Path(__file__).resolve().parents[2] / "data/raw/uci309_original.zip"
OUTPUT = HERE / "uci309_external_stress_2026_10_23.json"
PREDICTIONS = HERE / "uci309_external_stress_predictions_2026_10_23.csv"
PATTERN = re.compile(r"dataset_twosources_downsampled/(\d{3})_Et_([nLMH])_(CO|Me)_([nLMH])")


def load_trials():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == PROTOCOL["archive_sha256"]
    records = []
    with ZipFile(ARCHIVE) as z:
        names = sorted(name for name in z.namelist() if PATTERN.fullmatch(name))
        assert len(names) == 180
        for name in names:
            file_id, ethylene, gas, flow = PATTERN.fullmatch(name).groups()
            if flow == "n":
                continue
            with z.open(name) as stream:
                a = np.loadtxt(stream, delimiter=",", dtype=np.float64)
            t = a[:, 0]
            voltage = a[:, 3:11]
            assert a.shape[1] == 11 and np.isfinite(a).all()
            assert (voltage > 0).all() and (voltage < 3110).all()
            signal = np.log(10 * (3110 - voltage) / voltage)
            baseline = np.median(signal[(t >= 25) & (t < 45)], axis=0)
            curve = np.array([
                np.median(signal[(t >= start) & (t < start + 2)], axis=0) - baseline
                for start in np.arange(60, 100, 2)
            ])
            assert curve.shape == (20, 8) and np.isfinite(curve).all()
            records.append({
                "file": name, "id": file_id, "ethylene": ethylene, "gas": gas,
                "flow": flow, "y": int(gas == "CO"), "x": curve, "baseline": baseline,
            })
    assert len(records) == 144
    assert sum(r["y"] for r in records if r["flow"] == "L") == 24
    return records


def fit_cnn(train_x, train_y, test_x):
    """Twin pilot architecture, with four outputs changed to two."""
    mu = train_x.mean(axis=(0, 1), keepdims=True)
    sd = np.maximum(train_x.std(axis=(0, 1), keepdims=True), 0.03)

    def windows(x):
        x = np.clip((x - mu) / sd, -8, 8)
        return np.lib.stride_tricks.sliding_window_view(x, 3, axis=1).transpose(0, 1, 3, 2).reshape(len(x), 18, 24)

    a, b = windows(train_x), windows(test_x)
    shapes = ((24, 16), (16,), (32, 2), (2,))
    sizes = [int(np.prod(shape)) for shape in shapes]

    def unpack(p):
        offsets = np.cumsum([0, *sizes])
        return [p[offsets[i]:offsets[i + 1]].reshape(shape) for i, shape in enumerate(shapes)]

    def objective(p):
        w, bias, v, c = unpack(p)
        z = a @ w + bias
        h = np.maximum(z, 0)
        pooled = np.concatenate((h[:, :9].mean(axis=1), h[:, 9:].mean(axis=1)), axis=1)
        logits = pooled @ v + c
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.log(np.maximum(probs[np.arange(len(train_y)), train_y], 1e-15)).mean()
        loss += 0.02 * (np.square(w).sum() + np.square(v).sum()) / 2
        dz = probs
        dz[np.arange(len(train_y)), train_y] -= 1
        dz /= len(train_y)
        gv, gc = pooled.T @ dz + 0.02 * v, dz.sum(axis=0)
        dp = dz @ v.T
        dh = np.zeros_like(h)
        dh[:, :9] = dp[:, None, :16] / 9
        dh[:, 9:] = dp[:, None, 16:] / 9
        dpre = dh * (z > 0)
        gw = np.einsum("ntd,ntk->dk", a, dpre) + 0.02 * w
        gb = dpre.sum(axis=(0, 1))
        return loss, np.concatenate([q.ravel() for q in (gw, gb, gv, gc)])

    rng = np.random.default_rng(2026)
    init = np.concatenate([
        rng.normal(0, np.sqrt(2 / 24), shapes[0]).ravel(),
        np.full(shapes[1], 0.01),
        rng.normal(0, np.sqrt(2 / 32), shapes[2]).ravel(),
        np.zeros(shapes[3]),
    ])
    fit = minimize(objective, init, jac=True, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-7})
    w, bias, v, c = unpack(fit.x)
    h = np.maximum(b @ w + bias, 0)
    pooled = np.concatenate((h[:, :9].mean(axis=1), h[:, 9:].mean(axis=1)), axis=1)
    return (pooled @ v + c).argmax(axis=1), {"iterations": fit.nit, "converged": bool(fit.success)}


def rms_normalize(x):
    return x / np.maximum(np.sqrt(np.mean(np.square(x), axis=(1, 2), keepdims=True)), 1e-8)


def metrics(y, pred):
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "co_recall": float(matrix[1, 1] / matrix[1].sum()),
        "co_precision": float(matrix[1, 1] / matrix[:, 1].sum()) if matrix[:, 1].sum() else 0.0,
        "co_false_negatives": int(matrix[1, 0]),
        "false_co": int(matrix[0, 1]), "confusion_no_co_co": matrix.tolist(),
    }


def main():
    records = load_trials()
    x = np.array([r["x"] for r in records])
    y = np.array([r["y"] for r in records])
    flow = np.array([r["flow"] for r in records])
    ethylene = np.array([r["ethylene"] for r in records])
    train, test = np.isin(flow, ["M", "H"]), flow == "L"
    assert (train.sum(), test.sum(), y[train].sum(), y[test].sum()) == (96, 48, 48, 24)

    factors = (0.5, 0.75, 1.0)
    augmented = np.concatenate([x[train] * value for value in factors])
    augmented_y = np.tile(y[train], len(factors))
    model_inputs = {
        "cnn_raw": (x[train], y[train], x[test]),
        "cnn_amplitude": (augmented, augmented_y, x[test]),
        "cnn_rms": (rms_normalize(x[train]), y[train], rms_normalize(x[test])),
        "svm_raw": (x[train], y[train], x[test]),
        "svm_amplitude": (augmented, augmented_y, x[test]),
        "svm_rms": (rms_normalize(x[train]), y[train], rms_normalize(x[test])),
    }
    predictions, scores, fitting = {}, {}, {}
    for name, (xt, yt, xv) in model_inputs.items():
        if name.startswith("cnn"):
            pred, fitting[name] = fit_cnn(xt, yt, xv)
        else:
            model = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))
            model.fit(xt.reshape(len(xt), -1), yt)
            pred = model.predict(xv.reshape(len(xv), -1))
        predictions[name] = pred
        scores[name] = metrics(y[test], pred)
        scores[name]["ethylene_strata"] = {
            level: metrics(y[test][ethylene[test] == level], pred[ethylene[test] == level])
            for level in ("n", "L", "M", "H")
        }
        print(name, scores[name]["macro_f1"], scores[name]["confusion_no_co_co"], flush=True)

    control = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale"))
    baselines = np.array([r["baseline"] for r in records])
    control.fit(baselines[train], y[train])
    control_pred = control.predict(baselines[test])
    scores["pre_release_control"] = metrics(y[test], control_pred)
    print("pre_release_control", scores["pre_release_control"], flush=True)
    a = scores["cnn_amplitude"]
    passes = (
        a["macro_f1"] >= 0.8 and a["co_recall"] >= 0.9 and a["false_co"] <= 4
        and a["macro_f1"] >= max(scores["cnn_rms"]["macro_f1"], scores["svm_amplitude"]["macro_f1"]) - 0.05
        and scores["pre_release_control"]["balanced_accuracy"] < 0.65
    )
    with PREDICTIONS.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "ethylene", "gas", "flow", "truth_co", *predictions, "pre_release_control"])
        for i, r in enumerate(np.array(records, dtype=object)[test]):
            writer.writerow([
                r["file"], r["ethylene"], r["gas"], r["flow"], r["y"],
                *(int(predictions[name][i]) for name in predictions), int(control_pred[i]),
            ])
    OUTPUT.write_text(json.dumps({
        "protocol_file": str(HERE / "uci309_external_stress_protocol_2026_10_23.json"),
        "source_sha256": PROTOCOL["archive_sha256"], "train": int(train.sum()), "test": int(test.sum()),
        "scores": scores, "fit": fitting, "passes_predeclared_useful_complement": bool(passes),
    }, ensure_ascii=False, indent=2) + "\n")
    print("passes", passes, "saved", OUTPUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    ARCHIVE = parser.parse_args().archive
    main()
