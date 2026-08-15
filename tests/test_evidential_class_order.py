#!/usr/bin/env python
# coding=utf-8
"""
P0-2 类别顺序可复核测试（无需 GPU）。

覆盖 GPT 方案 3.2 的三条测试：
  1. Dirichlet 正类概率位于通道 1（正样本 -> p>0.5，负样本 -> p<0.5）
  2. 导出标签顺序（正样本在前，gt=1）与 AUC 方向
  3. 训练目标构造 make_target（pos -> [0,1]，neg -> [1,0]），
     从 eviddie_trainer.py 导入，避免训练/测试两套类别定义。

运行：python tests/test_evidential_class_order.py（仓库根目录）
"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# 允许导入仓库内模块
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'EviDDIE'))


def dirichlet_positive_probability(raw_output):
    """复刻导出脚本的评分：evidence=Softplus(fc_out); alpha=evidence+1; p=alpha[:,1]/S"""
    evidence = torch.nn.functional.softplus(raw_output)
    alpha = evidence + 1.0
    return (alpha / alpha.sum(dim=1, keepdim=True))[:, 1]


def test_positive_class_is_index_one():
    raw = torch.tensor([
        [-5.0, 5.0],   # 通道 1 更强 -> 应为正类
        [5.0, -5.0],   # 通道 0 更强 -> 应为负类
    ])
    p = dirichlet_positive_probability(raw)
    assert float(p[0]) > 0.5, f'positive sample scored {p[0]}'
    assert float(p[1]) < 0.5, f'negative sample scored {p[1]}'


def test_export_label_order():
    y = np.array([1, 1, 0, 0])   # 正样本在前（导出脚本：gt = [ones(n_pos), zeros(rest)]）
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
