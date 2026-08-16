#!/usr/bin/env python
# coding=utf-8
"""完整 EviDDIE 在训练器内部 dev 口径下的评估 (2026-08-16)。

目的：为消融训练曲线图提供完整模型的水平参考线。
口径与 AblationTrainer._eval_dev 完全一致（同一 dev_rows manifest），
保证与三条训练曲线可比。"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import numpy as np
import torch

from eviddie_args import read_options
from eviddie_train_ablation import AblationTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
PREFIXES = ['eviddie_new_s1', 'eviddie_new_s2', 'eviddie_new_s3',
            'eviddie_new_s4', 'eviddie_new_s5']


def main():
    rows = []
    for seed, prefix in zip(SEEDS, PREFIXES):
        args = read_options()
        args.dataset = 'dataset1'
        args.semantic = 'event_embedding2.json'
        args.seed = seed
        tr = AblationTrainer(args, train_seed=seed, prefix=prefix)
        # 注意：不能 reset_head —— 保留 checkpoint 原生 fc（完整模型头）
        acc, auroc, f1 = tr._eval_dev('full_evi', use_bsa=True)
        logging.info(f'seed={seed}: dev acc={acc:.4f} auroc={auroc:.4f} f1={f1:.4f}')
        rows.append({'train_seed': seed, 'dev_acc': acc,
                     'dev_auroc': auroc, 'dev_f1': f1})

    import csv
    out = 'results/full_evi_dev_internal.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['train_seed', 'dev_acc', 'dev_auroc', 'dev_f1'])
        w.writeheader()
        w.writerows(rows)
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
