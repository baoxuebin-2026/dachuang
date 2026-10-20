"""Check reconstructed per-file predictions against the published baselines."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    rows = read('uci361_recovered_predictions_2026-10-20.csv')
    one = {row['file']: row for row in rows if row['split'] == 'one'}
    assert len(one) == 192
    assert len(rows) == 960
    sources = (
        ('uci361_nested_device_predictions_2026-10-19.csv', None, 'aug_cnn'),
        ('uci361_tinycnn_predictions_2026-10-19.csv', 'low_held_board', 'raw_cnn'),
        ('uci361_three_scale_classical_predictions_2026-10-19.csv', 'temporal_logistic', 'aug_logistic'),
        ('uci361_three_scale_classical_predictions_2026-10-19.csv', 'temporal_rbf_svm', 'aug_svm'),
    )
    for filename, method, key in sources:
        baseline = read(filename)
        field = 'task' if filename.startswith('uci361_tinycnn') else 'method'
        baseline = [r for r in baseline if method is None or r[field] == method]
        assert len(baseline) == 192, (filename, len(baseline))
        disagreements = [r['file'] for r in baseline if one[r['file']][key] != r['prediction']]
        assert not disagreements, (filename, disagreements[:5])
        print('matched 192 individual predictions:', key)

    data = json.loads((ROOT / 'uci361_recovered_audit_2026-10-20.json').read_text())
    folds = data['one_device']
    expected = ('norm_cnn', 'aug_hybrid', 'aug_svm', 'aug_hybrid', 'aug_logistic')
    assert tuple(f['selected'] for f in folds) == expected
    assert [f['outer_scores']['norm_cnn']['co_tp'] for f in folds] == [6, 12, 12, 6, 6]
    assert abs(folds[0]['outer_scores']['norm_cnn']['macro_f1'] - 0.8303585049580473) < 1e-12
    assert abs(folds[4]['outer_scores']['aug_logistic']['macro_f1'] - 2 / 3) < 1e-12
    fixed_augmented = [0.9386253842775583, 0.9580419580419581, 1.0, 1.0, 1.0]
    fixed_hybrid = [0.9791304347826086, 0.9791304347826086, 1.0, 1.0, 1.0]
    for fold, augmented, hybrid in zip(folds, fixed_augmented, fixed_hybrid):
        assert abs(fold['outer_scores']['aug_cnn']['macro_f1'] - augmented) < 1e-12
        assert abs(fold['outer_scores']['aug_hybrid']['macro_f1'] - hybrid) < 1e-12
    print('matched retrospective choices, normalized control and worst device')

    pairs = defaultdict(list)
    for row in rows:
        if row['split'] == 'two':
            pairs[row['held_devices']].append(row)
    assert len(pairs) == 10
    expected_pair_means = {
        'aug_cnn': 0.9641561386202427,
        'aug_logistic': 0.9383089178956222,
        'aug_svm': 0.951919299732358,
        'norm_cnn': 0.9527982382453827,
        'raw_cnn': 0.6777057966295783,
        'aug_hybrid': 0.9798367168863642,
    }
    for candidate, expected in expected_pair_means.items():
        values = [f1_score([r['truth'] for r in group], [r[candidate] for r in group],
                           labels=['CO', 'Me', 'Ea', 'Ey'], average='macro')
                  for group in pairs.values()]
        assert abs(sum(values) / len(values) - expected) < 1e-12, candidate
    print('matched ten overlapping pair summaries (not independent replicates)')

    duplicate = json.loads((ROOT / 'uci361_recovered_duplicate_control_2026-10-20.json').read_text())
    for method, expected in {'cnn': 0.6785934212810141,
                             'logistic': 0.8013990956464534,
                             'svm': 0.7467189426154526}.items():
        assert abs(duplicate['pair_mean_f1'][method] - expected) < 1e-12
    print('matched three-copy control for all three models')


if __name__ == '__main__':
    main()
