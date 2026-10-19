"""Nested device-based model selection for the exploratory UCI 361 CNN."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import uci361_tinycnn_pilot as conv

ROOT = Path(__file__).resolve().parent
SCHEMES = {
    'none': (1.0,),
    'three_scales': (0.5, 0.75, 1.0),
    'four_scales': (0.25, 0.5, 0.75, 1.0),
}
CO_BIASES = (-0.5, 0.0, 0.5)


def augmentation(data, labels, factors):
    return np.concatenate([data * a for a in factors]), np.tile(labels, len(factors))


def predict(logits, co_bias):
    scores = logits.copy()
    scores[:, 0] += co_bias
    return scores.argmax(axis=1)


def score(y, yp):
    mat = confusion_matrix(y, yp, labels=range(4))
    n_co = mat[0].sum()
    n_other = mat[1:].sum()
    return {
        'macro_f1': float(f1_score(y, yp, labels=range(4), average='macro', zero_division=0)),
        'co_recall': float(mat[0, 0] / n_co),
        'co_false_positive_rate': float(mat[1:, 0].sum() / n_other),
        'co_false_negative_count': int(n_co - mat[0, 0]),
        'co_false_positive_count': int(mat[1:, 0].sum()),
        'methane_recall': float(mat[1, 1] / mat[1].sum()),
        'confusion': mat.tolist(),
    }


def rank(record):
    metrics = record['metrics']
    eligible = (metrics['co_recall'] >= 0.95 and
                metrics['co_false_positive_rate'] <= 0.05)
    # The constraint and tie order are fixed before this nested run.
    return (int(eligible), metrics['macro_f1'], metrics['co_recall'],
            -metrics['co_false_positive_rate'],
            -len(SCHEMES[record['scheme']]), -abs(record['co_bias']),
            record['scheme'])


def main():
    conv.gradient_check()
    x, y, boards, levels, names = conv.get_trials()
    all_pred = np.full(len(y), -1)
    folds = []
    for held in range(1, 6):
        inner_boards = [b for b in range(1, 6) if b != held]
        responses = {}
        validation_truth = None
        for scheme, factors in SCHEMES.items():
            logits_parts = []
            truth_parts = []
            for valid_board in inner_boards:
                inner_train = (boards != held) & (boards != valid_board) & (levels >= 7)
                inner_val = (boards == valid_board) & (levels >= 4) & (levels <= 6)
                xx, yy = augmentation(x[inner_train], y[inner_train], factors)
                logits, fit = conv.train_predict(xx, yy, x[inner_val], return_logits=True)
                logits_parts.append(logits)
                truth_parts.append(y[inner_val])
                print('outer',held,'inner',valid_board,scheme,'n',int(inner_val.sum()),
                      'iterations',fit['iterations'],flush=True)
            responses[scheme] = np.concatenate(logits_parts)
            truth = np.concatenate(truth_parts)
            if validation_truth is None:
                validation_truth = truth
            else:
                assert np.array_equal(validation_truth, truth)
        candidates = []
        for scheme, logits in responses.items():
            for bias in CO_BIASES:
                candidates.append({'scheme': scheme, 'co_bias': bias,
                    'metrics': score(validation_truth, predict(logits, bias))})
        chosen = max(candidates, key=rank)
        final_train = (boards != held) & (levels >= 4)
        final_test = (boards == held) & (levels <= 3)
        xx, yy = augmentation(x[final_train], y[final_train], SCHEMES[chosen['scheme']])
        logits, fit = conv.train_predict(xx, yy, x[final_test], return_logits=True)
        all_pred[final_test] = predict(logits, chosen['co_bias'])
        test_metrics = score(y[final_test], all_pred[final_test])
        folds.append({'held_board': held, 'validation_n': len(validation_truth),
                      'selection': chosen, 'candidates': candidates,
                      'outer_test_n': int(final_test.sum()), 'outer_test': test_metrics,
                      'fit': fit})
        print('outer',held,'chosen',chosen['scheme'],chosen['co_bias'],
              'validation',chosen['metrics']['macro_f1'],
              'test',test_metrics['macro_f1'],flush=True)
        checkpoint=ROOT/'uci361_nested_selection_checkpoint_2026-10-19.json'
        checkpoint.write_text(json.dumps({'completed_outer_folds':folds},ensure_ascii=False,indent=2)+'\n')
    tested=all_pred!=-1
    final = {'n':int(tested.sum()),**score(y[tested],all_pred[tested]),
             'by_board_f1':{str(k):float(f1_score(y[tested&(boards==k)],
              all_pred[tested&(boards==k)],labels=range(4),average='macro',zero_division=0)) for k in range(1,6)}}
    out=ROOT/'uci361_nested_device_selection_exploratory_2026-10-19.json'
    out.write_text(json.dumps({'protocol':
          'Exploratory nested selection after the same devices had already been inspected. Outer held board ranks 1-3; inner training other boards ranks 7-10 and inner held-board validation ranks 4-6; final training other boards ranks 4-10. Three fixed amplitude regimes and CO biases chosen within training devices only.',
          'schemes':SCHEMES,'co_biases':CO_BIASES,'outer_folds':folds,'overall':final},
          ensure_ascii=False,indent=2)+'\n')
    rows=[{'file':names[i],'board':int(boards[i]),'level':int(levels[i]),
           'truth':conv.LABELS[y[i]],'prediction':conv.LABELS[all_pred[i]]} for i in np.where(tested)[0]]
    with (ROOT/'uci361_nested_device_predictions_2026-10-19.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print('saved',out,flush=True)
    print('overall',final,flush=True)


if __name__=='__main__':
    main()
