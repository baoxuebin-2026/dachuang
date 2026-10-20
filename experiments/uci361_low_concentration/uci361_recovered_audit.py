"""Reconstruct the exploratory device-level comparisons from the source archive.

This script was recovered after the original local scratch directory expired.
The methods and the selection policy were chosen after all five devices had
already been examined. Its outer folds are retrospective, not blind tests.
"""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import uci361_tinycnn_pilot as cnn


ROOT = Path(__file__).resolve().parent
LABELS = cnn.LABELS
FACTORS = (0.5, 0.75, 1.0)
CANDIDATES = ('aug_cnn', 'aug_logistic', 'aug_svm', 'norm_cnn', 'raw_cnn', 'aug_hybrid')


def normalize(x):
    """Per-record global RMS normalization with a smooth 0.05 stabilizer."""
    rms = np.sqrt(np.mean(np.square(x), axis=(1, 2), keepdims=True))
    return x / np.sqrt(np.square(rms) + 0.05 ** 2)


def metrics(truth, pred):
    mat = confusion_matrix(truth, pred, labels=LABELS)
    return {
        'n': len(truth),
        'macro_f1': float(f1_score(truth, pred, labels=LABELS, average='macro', zero_division=0)),
        'co_tp': int(mat[0, 0]), 'co_n': int(mat[0].sum()),
        'nonco_as_co': int(mat[1:, 0].sum()),
        'co_recall': float(mat[0, 0] / mat[0].sum()),
        'nonco_fp_rate': float(mat[1:, 0].sum() / mat[1:, :].sum()),
        'confusion': mat.tolist(),
    }


def predict_all(x, xn, y, train, test):
    augmented = np.concatenate([x[train] * factor for factor in FACTORS])
    labels = np.tile(y[train], len(FACTORS))
    result = {}
    result['aug_cnn'], _ = cnn.train_predict(augmented, labels, x[test])
    result['raw_cnn'], _ = cnn.train_predict(x[train], y[train], x[test])
    result['norm_cnn'], _ = cnn.train_predict(xn[train], y[train], xn[test])
    for name, classifier in (
        ('aug_logistic', LogisticRegression(max_iter=3000, C=1)),
        ('aug_svm', SVC(C=1, gamma='scale')),
    ):
        pipe = make_pipeline(StandardScaler(), classifier)
        pipe.fit(augmented.reshape(len(augmented), -1), labels)
        result[name] = pipe.predict(x[test].reshape(int(test.sum()), -1))
    a, b = result['aug_cnn'], result['aug_logistic']
    result['aug_hybrid'] = np.where((a == 0) | (b == 0), b, a)
    return result


def score_candidates(pair_predictions, boards, ranks, names, truth, outer):
    scores = {}
    for candidate in CANDIDATES:
        inner = []
        for held in range(1, 6):
            if held == outer:
                continue
            pair = tuple(sorted((outer, held)))
            ix = np.flatnonzero((boards == held) & (ranks <= 3))
            predictions = [pair_predictions[(pair, names[i])][candidate] for i in ix]
            inner.append(metrics([truth[i] for i in ix], predictions))
        scores[candidate] = {
            'mean_inner_f1': float(np.mean([v['macro_f1'] for v in inner])),
            'min_inner_f1': float(min(v['macro_f1'] for v in inner)),
            'min_inner_co_recall': float(min(v['co_recall'] for v in inner)),
            'mean_inner_nonco_fp_rate': float(np.mean([v['nonco_fp_rate'] for v in inner])),
            'per_inner_board': inner,
        }
    eligible = [c for c in CANDIDATES if scores[c]['min_inner_co_recall'] >= 0.95
                and scores[c]['mean_inner_nonco_fp_rate'] <= 0.05]
    if not eligible:
        eligible = sorted(CANDIDATES, key=lambda c: (scores[c]['min_inner_co_recall'],
                          -scores[c]['mean_inner_nonco_fp_rate']), reverse=True)[:1]
    key = lambda c: (scores[c]['mean_inner_f1'], scores[c]['min_inner_f1'], -CANDIDATES.index(c))
    return scores, eligible, max(eligible, key=key)


def main():
    cnn.gradient_check()
    x, y, boards, ranks, names = cnn.get_trials()
    xn = normalize(x)
    truth = [LABELS[i] for i in y]
    one_predictions = {}
    two_predictions = {}
    output_rows = []
    folds = []

    for held in range(1, 6):
        train = (boards != held) & (ranks >= 4)
        test = (boards == held) & (ranks <= 3)
        pred = predict_all(x, xn, y, train, test)
        for i, j in enumerate(np.flatnonzero(test)):
            row = {method: LABELS[values[i]] for method, values in pred.items()}
            one_predictions[names[j]] = row
            output_rows.append({'split': 'one', 'held_devices': str(held), 'file': names[j],
                                'truth': truth[j], **row})
        print('one-device', held, flush=True)

    for pair in itertools.combinations(range(1, 6), 2):
        train = (~np.isin(boards, pair)) & (ranks >= 4)
        test = np.isin(boards, pair) & (ranks <= 3)
        pred = predict_all(x, xn, y, train, test)
        for i, j in enumerate(np.flatnonzero(test)):
            row = {method: LABELS[values[i]] for method, values in pred.items()}
            two_predictions[(pair, names[j])] = row
            output_rows.append({'split': 'two', 'held_devices': '-'.join(map(str, pair)),
                                'file': names[j], 'truth': truth[j], **row})
        print('two-device', pair, flush=True)

    for outer in range(1, 6):
        scores, eligible, selected = score_candidates(two_predictions, boards, ranks, names, truth, outer)
        ix = np.flatnonzero((boards == outer) & (ranks <= 3))
        outer_scores = {c: metrics([truth[i] for i in ix],
                        [one_predictions[names[i]][c] for i in ix]) for c in CANDIDATES}
        folds.append({'held_device': outer, 'selected': selected,
                      'eligible_candidates': eligible, 'inner_candidate_scores': scores,
                      'outer_scores': outer_scores})
        print('selected', outer, selected, outer_scores[selected]['macro_f1'], flush=True)

    with (ROOT / 'uci361_recovered_predictions_2026-10-20.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        'source_sha256': 'e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167',
        'scope': 'Reconstructed retrospective audit, NOT new independent validation. Candidate set and selection rule were specified after outcomes were known.',
        'input': '20 x 8 log-response differences; train other units ranks 4-10; test held units ranks 1-3',
        'normalization': 'global per-record RMS, denominator sqrt(rms^2 + 0.05^2)',
        'factors': FACTORS, 'candidate_order': CANDIDATES,
        'selection': 'min inner board CO recall >=0.95 and mean non-CO-as-CO rate <=0.05, then largest mean inner F1',
        'one_device': folds,
    }
    (ROOT / 'uci361_recovered_audit_2026-10-20.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('saved', ROOT / 'uci361_recovered_audit_2026-10-20.json', flush=True)


if __name__ == '__main__':
    main()
