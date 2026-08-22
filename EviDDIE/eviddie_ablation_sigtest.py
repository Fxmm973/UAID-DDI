#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy import stats

CSV = 'results/predictions/predictions_eviddie_new_ablation.csv'
df = pd.read_csv(CSV)
settings = ['common', 'fewer', 'rare']
variants = ['Softmax baseline', 'EviDDIE w/o BSA', 'EviDDIE w/o EVI']
METRICS = {'AUROC': 'auroc', 'F1': 'f1', 'ACC': 'acc', 'AUPR': 'aupr'}


def per_seed(g):
    out = {}
    for seed, sg in g.groupby('train_seed'):
        yt, yp = sg['y_true'], sg['prob']
        if yt.nunique() < 2:
            continue
        sp = (yp >= 0.5).astype(int)
        out[seed] = {
            'auroc': metrics.roc_auc_score(yt, yp),
            'f1': metrics.f1_score(yt, sp, zero_division=0),
            'acc': metrics.accuracy_score(yt, sp),
            'aupr': metrics.average_precision_score(yt, yp),
        }
    return out


rows = []
print(f'{"setting":<8}{"metric":<8}{"variant":<22}{"full":<10}{"var":<10}{"diff":<9}{"p(paired-t)":<13}sig')
print('-' * 90)
for st in settings:
    full_ps = per_seed(df[(df.setting == st) & (df.method == 'EviDDIE')])
    for v in variants:
        var_ps = per_seed(df[(df.setting == st) & (df.method == v)])
        for mname, mkey in METRICS.items():
            fv = [full_ps[s][mkey] for s in full_ps]
            vv = [var_ps[s][mkey] for s in var_ps]
            t, p = stats.ttest_rel(vv, fv)
            sig = '**' if p < 0.05 else ('*' if p < 0.10 else 'ns')
            print(f'{st:<8}{mname:<8}{v:<22}{np.mean(fv):<10.4f}{np.mean(vv):<10.4f}'
                  f'{np.mean(vv) - np.mean(fv):<+9.4f}{p:<13.3f}{sig}')
            rows.append({'setting': st, 'metric': mname, 'variant': v,
                         'full_mean': np.mean(fv), 'variant_mean': np.mean(vv),
                         'diff': np.mean(vv) - np.mean(fv), 'p_paired_t': p})
    print('-' * 90)

pd.DataFrame(rows).to_csv('results/ablation_sigtest.csv', index=False)
print('\nSaved -> results/ablation_sigtest.csv')
