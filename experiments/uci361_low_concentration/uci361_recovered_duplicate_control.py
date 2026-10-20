"""Compare the three-scale training set against three exact copies of each trial.

The ten two-device splits overlap. Their mean scores are descriptive only.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import uci361_tinycnn_pilot as cnn


ROOT = Path(__file__).resolve().parent


def main():
    cnn.gradient_check()
    x, y, boards, ranks, _ = cnn.get_trials()
    rows = []
    for pair in itertools.combinations(range(1, 6), 2):
        train = (~np.isin(boards, pair)) & (ranks >= 4)
        test = np.isin(boards, pair) & (ranks <= 3)
        xx = np.concatenate([x[train]] * 3)
        yy = np.tile(y[train], 3)
        predictions = {}
        predictions['cnn'], _ = cnn.train_predict(xx, yy, x[test])
        for name, model in (
            ('logistic', LogisticRegression(max_iter=3000, C=1)),
            ('svm', SVC(C=1, gamma='scale')),
        ):
            pipe = make_pipeline(StandardScaler(), model)
            pipe.fit(xx.reshape(len(xx), -1), yy)
            predictions[name] = pipe.predict(x[test].reshape(int(test.sum()), -1))
        row = {'pair': list(pair), 'test_files': int(test.sum()),
               'f1': {method: float(f1_score(y[test], values, labels=range(4),
                          average='macro', zero_division=0))
                      for method, values in predictions.items()}}
        rows.append(row)
        print('pair', pair, row['f1'], flush=True)
    result = {'scope': 'Retrospective check of three exact duplicates versus amplitude perturbation; overlapping pair folds are not independent.',
              'pairs': rows,
              'pair_mean_f1': {method: float(np.mean([r['f1'][method] for r in rows]))
                               for method in ('cnn', 'logistic', 'svm')}}
    (ROOT / 'uci361_recovered_duplicate_control_2026-10-20.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['pair_mean_f1'], flush=True)


if __name__ == '__main__':
    main()
