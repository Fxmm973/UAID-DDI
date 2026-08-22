#!/usr/bin/env python
# coding=utf-8
import csv
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
VARIANTS = ['Softmax baseline', 'EviDDIE w/o EVI', 'EviDDIE w/o BSA', 'EviDDIE']
DATA_CHECKS = [
    'dataset1/train_tasks.json',
    'dataset1/DRKG_TransE_entity.npy',
    'dataset1/event_embedding2.json',
    'neg_manifests/manifest_hashes.json',
    'models/dataset1/pharddie_best.pt',
    'models/dataset1/bestmodels_G',
]


def ckpt_exists(seed):
    return (os.path.exists(f'models/dataset1/eviddie_0shot_seed{seed}bestmodel')
            or os.path.exists(f'models/dataset1/eviddie_0shot_seed{seed}/bestmodel'))


def step(cmd, name):
    print(f'\n===== {name} =====\n  {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, cwd=BASE)
    if r.returncode != 0:
        raise SystemExit(f'{name} failed (exit {r.returncode}); see the log above')


def ece15(conf, pred, lab):
    b = np.linspace(0, 1, 16)
    e = 0.0
    for i in range(15):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum() > 0:
            e += m.sum() / len(conf) * np.abs((pred[m] == lab[m]).mean() - conf[m].mean())
    return e


def cal_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    ece = ece15(conf, pred, y)
    brier = np.mean((p - y) ** 2)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    nll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    hc = conf > 0.9
    hce = (pred[hc] != y[hc]).mean() if hc.sum() > 0 else 0.0
    return ece, brier, nll, hce


def disc_metrics(y, p, ev):
    pred = (p >= 0.5).astype(int)
    auc = metrics.roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    acc = metrics.accuracy_score(y, pred)
    f1s = []
    for e in np.unique(ev):
        m = ev == e
        if len(np.unique(y[m])) > 1:
            f1s.append(metrics.f1_score(y[m], pred[m], zero_division=0))
    f1 = np.mean(f1s) if f1s else float('nan')
    return auc, acc, f1


def aggregate():
    csv_path = 'results/predictions/predictions_eviddie_new_ablation.csv'
    df = pd.read_csv(csv_path)
    print('\n===== Aggregated results (mean +/- SD over 5 training seeds) =====')
    rows = []
    for method in VARIANTS:
        sub = df[df.method == method]
        cal_rows, disc_rows = {}, {}
        for seed, g in sub.groupby('train_seed'):
            e, br, n, h = cal_metrics(g['y_true'].values, g['prob'].values)
            cal_rows.setdefault(seed, (e, br, n, h))
            for setting in ['fewer', 'rare']:
                s = g[g.setting == setting]
                disc_rows.setdefault((seed, setting),
                                     disc_metrics(s['y_true'].values, s['prob'].values,
                                                  s['event_type'].values))
        c = np.array(list(cal_rows.values()))
        line = (f'{method:16s} | ECE {c[:,0].mean():.4f}+/-{c[:,0].std():.4f} '
                f'Brier {c[:,1].mean():.4f}+/-{c[:,1].std():.4f} '
                f'NLL {c[:,2].mean():.4f}+/-{c[:,2].std():.4f} '
                f'HCE {c[:,3].mean():.4f}+/-{c[:,3].std():.4f}')
        print(line)
        rows.append([method, 'calibration'] + [f'{x.mean():.4f}+/-{x.std():.4f}' for x in c.T])
        for setting in ['fewer', 'rare']:
            d = np.array([disc_rows[(s, setting)] for s in SEEDS])
            print(f'  {setting:6s} | AUROC {d[:,0].mean():.4f}+/-{d[:,0].std():.4f} '
                  f'ACC {d[:,1].mean():.4f}+/-{d[:,1].std():.4f} '
                  f'macro-F1 {d[:,2].mean():.4f}+/-{d[:,2].std():.4f}')
            rows.append([method, setting] + [f'{x.mean():.4f}+/-{x.std():.4f}' for x in d.T])
    with open('results/eviddie_p0_2_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'setting', 'auc', 'acc', 'f1', 'ece', 'brier', 'nll', 'hce'])
        w.writerows(rows)
    print('\nSaved to results/eviddie_p0_2_results.csv')


def main():
    missing = [p for p in DATA_CHECKS if not os.path.exists(os.path.join(BASE, p))]
    if missing:
        raise SystemExit(f'Missing data/backbone files: {missing}')

    need_heads = [p for p in ['models/dataset1/fc_softmax.pt',
                              'models/dataset1/fc_evi_no_evi.pt',
                              'models/dataset1/fc_w_o BSA.pt',
                              'models/dataset1/linear_proj_wo_BSA.pt']
                  if not os.path.exists(os.path.join(BASE, p))]
    if need_heads:
        step([sys.executable, 'eviddie_train_ablation.py'],
             'Step 1/4: train the four variant heads (frozen backbone)')
    else:
        print('[SKIP] variant heads already present')

    for seed in SEEDS:
        if ckpt_exists(seed):
            print(f'[SKIP] seed {seed} checkpoint exists')
            continue
        step([sys.executable, 'eviddie_trainer.py',
              '--dataset', 'dataset1', '--prefix', 'dataset1/eviddie_0shot',
              '--seed', str(seed), '--max_batches', '20000'],
             f'Step 2/4: train full EviDDIE seed {seed}')
    for seed in SEEDS:
        if not ckpt_exists(seed):
            raise SystemExit(f'seed {seed} checkpoint still missing after training')

    legacy = 'results/predictions/predictions_dataset1_zero_shot_variants.csv'
    if os.path.exists(legacy) and not os.path.exists(legacy.replace('.csv', '_legacy.csv')):
        os.rename(legacy, legacy.replace('.csv', '_legacy.csv'))
        print('[BACKUP] legacy CSV renamed to ..._zero_shot_variants_legacy.csv')
    step([sys.executable, 'eviddie_export_zs_v2.py'],
         'Step 3/4: export four variants x five seeds with fixed manifests')

    print('Step 4/4: aggregate calibration + discrimination')
    aggregate()
    print('\nDone.')


if __name__ == '__main__':
    main()
