#!/usr/bin/env python
# coding=utf-8
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'EviDDIE'))


def dirichlet_positive_probability(raw_output):
    evidence = torch.nn.functional.softplus(raw_output)
    alpha = evidence + 1.0
    return (alpha / alpha.sum(dim=1, keepdim=True))[:, 1]


def test_positive_class_is_index_one():
    raw = torch.tensor([
        [-5.0, 5.0],
        [5.0, -5.0],
    ])
    p = dirichlet_positive_probability(raw)
    assert float(p[0]) > 0.5, f'positive sample scored {p[0]}'
    assert float(p[1]) < 0.5, f'negative sample scored {p[1]}'


def test_export_label_order():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc_score(y, p) == 1.0


def test_evidential_targets():
    from eviddie_trainer import make_target
    pos = make_target('pos', batch_size=3)
    neg = make_target('neg', batch_size=3)
    assert torch.equal(pos, torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))
    assert torch.equal(neg, torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))


if __name__ == '__main__':
    tests = [
        ('test_positive_class_is_index_one', test_positive_class_is_index_one),
        ('test_export_label_order', test_export_label_order),
        ('test_evidential_targets', test_evidential_targets),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {name}: {e}')
    print('ALL TESTS PASSED' if failed == 0 else f'{failed} TEST(S) FAILED')
    sys.exit(1 if failed else 0)
