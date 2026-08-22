#!/usr/bin/env python
# coding=utf-8
import csv
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

PH_CSV = 'PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv'
RR_TXT = 'PharDDIE/results/rareddie_seed_{seed}.txt'
OUT = 'PharDDIE/results/paired_diff_PharDDIE_RareDDIE.csv'
SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]


def pharddie_per_seed():
    df = pd.read_csv(PH_CSV)
    rare = df[df.setting == 'rare']
    out = {1: {}, 5: {}}
    for shot in (1, 5):
        for seed, sg in rare[rare.shot == shot].groupby('train_seed'):
            sp = (sg['prob'] >= 0.5).astype(int)
            f1s = []
            for ev, eg in sg.groupby('event_type'):
                f1s.append(metrics.f1_score(eg['y_true'],
                                            (eg['prob'] >= 0.5).astype(int),
                                            zero_division=0))
            out[shot][seed] = {
                'AUC': metrics.roc_auc_score(sg['y_true'], sg['prob']),
                'ACC': metrics.accuracy_score(sg['y_true'], sp),
                'F1': float(np.mean(f1s)),
            }
    return out


def rareddie_per_seed():
    out = {1: {}, 5: {}}
    for seed in SEEDS:
        txt = open(RR_TXT.format(seed=seed)).read()
        for shot in (1, 5):
            m = re.search(rf'{shot}-shot test2: AUC=([\d.]+) ACC=([\d.]+) F1=([\d.]+)', txt)
            out[shot][seed] = {
                'AUC': float(m.group(1)),
                'ACC': float(m.group(2)),
                'F1': float(m.group(3)),
            }
    return out


def main():
    ph = pharddie_per_seed()
    rr = rareddie_per_seed()
    t = stats.t.ppf(0.975, 4)
    rows = []
    print('Paired difference PharDDIE - RareDDIE (rare/test2, per-seed paired)')
    for shot in (1, 5):
        print(f'--- {shot}-shot ---')
        for k in ('AUC', 'ACC', 'F1'):
            diffs = np.array([ph[shot][s][k] - rr[shot][s][k] for s in SEEDS])
            m = diffs.mean()
            se = diffs.std(ddof=1) / np.sqrt(5)
            lo, hi = m - t * se, m + t * se
            print(f'{k}: mean={m:+.4f} 95%CI=[{lo:+.4f}, {hi:+.4f}] '
                  f'per-seed={np.round(diffs, 4).tolist()}')
            for s, d in zip(SEEDS, diffs):
                rows.append({'shot': shot, 'metric': k, 'seed': s,
                             'pharddie': ph[shot][s][k], 'rareddie': rr[shot][s][k],
                             'paired_diff': d, 'mean_diff': m,
                             'ci95_low': lo, 'ci95_high': hi})
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[saved] {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
