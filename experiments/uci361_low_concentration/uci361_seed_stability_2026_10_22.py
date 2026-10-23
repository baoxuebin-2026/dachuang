"""Retrospective seed stress, frozen before running on the inspected UCI361 archive.

Complete held-device, ranks 4..10 train, ranks 1..3 test, 192 unique exposure
files. Only tiny CNN initialization changes (seeds 2023..2027), and maxiter
remains 400. Three-scale amplitude preprocessing and architecture are fixed.

Failure criterion for a stable *CNN headline*: fewer than four of five seeds
achieve at least 185/192 correct and 48/48 CO correct. Even a pass cannot
establish a meaningful advantage over the 185/192 augmented SVM; report the
entire range and CO false positives. This is not independent validation.
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
    outputs = {}
    for seed in range(2023, 2028):
        pred = np.full(len(y), -1)
        per_board = {}
        converged = {}
        for held in range(1, 6):
            tr = (boards != held) & (ranks >= 4)
            te = (boards == held) & (ranks <= 3)
            training = np.concatenate([x[tr] * a for a in (0.5, 0.75, 1.0)])
            targets = np.tile(y[tr], 3)
            pred[te], fit = cnn.train_predict(training, targets, x[te], seed=seed)
            converged[str(held)] = bool(fit["converged"])
            per_board[str(held)] = float(f1_score(y[te], pred[te], labels=range(4), average="macro"))
            print("seed", seed, "device", held, "F1", per_board[str(held)], flush=True)
        sel = pred >= 0
        assert sel.sum() == len({names[i] for i in np.flatnonzero(sel)}) == 192
        mat = confusion_matrix(y[sel], pred[sel], labels=range(4))
        outputs[str(seed)] = {
            "macro_f1": float(f1_score(y[sel], pred[sel], labels=range(4), average="macro")),
            "correct": int(np.sum(y[sel] == pred[sel])),
            "co_correct": int(mat[0, 0]),
            "non_co_as_co": int(mat[1:, 0].sum()),
            "per_board_macro_f1": per_board,
            "all_converged": all(converged.values()),
        }
    result = {
        "scope": "Retrospective UCI361, not independent validation",
        "fixed_protocol": "five leave-board-out folds; source ranks 4..10; target ranks 1..3; three-scale CNN; maxiter=400",
        "seeds": outputs,
        "stable_headline": sum(v["correct"] >= 185 and v["co_correct"] == 48 for v in outputs.values()) >= 4,
        "comparison_caveat": "The augmented SVM is 185/192 and has fewer false CO alerts at seed 2026; no substantial CNN superiority claim.",
    }
    dest = ROOT / "uci361_seed_stability_2026_10_22.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("RESULT", {k: {q: v[q] for q in ("macro_f1", "correct", "co_correct", "non_co_as_co")} for k, v in outputs.items()}, flush=True)


if __name__ == "__main__":
    main()
