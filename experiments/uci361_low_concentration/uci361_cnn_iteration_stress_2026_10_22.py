"""Frozen before running: optimizer-cap sensitivity on the inspected UCI361 data.

Same source-only five-device, high-to-low 192-file protocol and identical
architecture/initial seed. Change ONLY the L-BFGS-B maximum from 160 to 400.
Compare three-factor amplitude CNN with per-file RMS normalized CNN. If
amplitude loses its pooled four-class macro F1 advantage, or has fewer CO
correct than normalized CNN, the draft's claimed ordering is not robust to
this optimization setting. Report both positive and negative outcomes. This
is retrospective sensitivity, not independent validation.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import uci361_tinycnn_pilot as cnn


ROOT = Path(__file__).resolve().parent
original_minimize = cnn.minimize


def more_iterations(*args, **kwargs):
    kwargs["options"] = {**kwargs.get("options", {}), "maxiter": 400}
    return original_minimize(*args, **kwargs)


def main():
    cnn.minimize = more_iterations
    x, y, boards, ranks, names = cnn.get_trials()
    rms = np.sqrt(np.mean(np.square(x), axis=(1, 2), keepdims=True))
    normalized = x / np.sqrt(np.square(rms) + 0.05**2)
    results = {}
    for arm in ("three_scale", "per_record_rms"):
        pred = np.full(len(y), -1)
        fits = {}
        per_board = {}
        for held in range(1, 6):
            tr = (boards != held) & (ranks >= 4)
            te = (boards == held) & (ranks <= 3)
            if arm == "three_scale":
                training = np.concatenate([x[tr] * a for a in (0.5, 0.75, 1.0)])
                targets = np.tile(y[tr], 3)
                testing = x[te]
            else:
                training = normalized[tr]
                targets = y[tr]
                testing = normalized[te]
            pred[te], fits[str(held)] = cnn.train_predict(training, targets, testing, seed=2026)
            per_board[str(held)] = float(f1_score(y[te], pred[te], labels=range(4), average="macro"))
            print(arm, held, per_board[str(held)], fits[str(held)], flush=True)
        sel = pred >= 0
        assert sel.sum() == 192
        assert len(set(names[i] for i in np.flatnonzero(sel))) == 192
        mat = confusion_matrix(y[sel], pred[sel], labels=range(4))
        results[arm] = {
            "macro_f1": float(f1_score(y[sel], pred[sel], labels=range(4), average="macro")),
            "correct": int((y[sel] == pred[sel]).sum()),
            "co_correct": int(mat[0, 0]),
            "non_co_as_co": int(mat[1:, 0].sum()),
            "per_board_macro_f1": per_board,
            "confusion": mat.tolist(),
            "fit": fits,
        }
    a, b = results["three_scale"], results["per_record_rms"]
    result = {
        "scope": "Retrospective UCI361; not independent validation",
        "iteration_cap": 400,
        "source_sha256": "e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167",
        "models": results,
        "draft_ordering_robust": a["macro_f1"] > b["macro_f1"] and a["co_correct"] >= b["co_correct"],
    }
    dest = ROOT / "uci361_cnn_iteration_stress_2026_10_22.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("RESULT", {k: {q: v[q] for q in ("macro_f1", "correct", "co_correct", "non_co_as_co")} for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    main()
