#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

CSV_PATH = 'results/predictions/predictions_eviddie_new_ablation.csv'
OUT_PNG = 'EviDDIE_Ablation_Study_new.png'
OUT_CSV = 'results/ablation_summary_eviddie_new.csv'

METHOD_ORDER = ['Softmax baseline', 'EviDDIE w/o BSA', 'EviDDIE w/o EVI', 'EviDDIE']
SETTING_ORDER = ['common', 'fewer', 'rare']
SETTING_LABELS = {'common': 'Common\n(dev)', 'fewer': 'Fewer\n(test)', 'rare': 'Rare\n(test2)'}

COLORS = {'Softmax baseline': '#999999', 'EviDDIE w/o BSA': '#FF9800',
          'EviDDIE w/o EVI': '#2196F3', 'EviDDIE': '#D32F2F'}
MARKERS = {'Softmax baseline': 's', 'EviDDIE w/o BSA': 'D',
           'EviDDIE w/o EVI': '^', 'EviDDIE': 'o'}
LINESTYLES = {'Softmax baseline': ':', 'EviDDIE w/o BSA': (0, (2, 3)),
              'EviDDIE w/o EVI': '--', 'EviDDIE': '-'}

METRICS = ['auroc', 'f1', 'acc', 'aupr']


def per_seed_metrics(syt, syp):
    if syt.nunique() < 2:
        return None
    sp = (syp >= 0.5).astype(int)
    return {
        'auroc': metrics.roc_auc_score(syt, syp),
        'f1': metrics.f1_score(syt, sp, zero_division=0),
        'acc': metrics.accuracy_score(syt, sp),
        'aupr': metrics.average_precision_score(syt, syp),
    }


def main():
    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {len(df)} rows | methods: {sorted(df.method.unique())} '
          f'| train_seeds: {sorted(df.train_seed.unique())} '
          f'| settings: {sorted(df.setting.unique())}')

    results = {}
    rows = []
    for setting in SETTING_ORDER:
        for method in METHOD_ORDER:
            g = df[(df.setting == setting) & (df.method == method)]
            seed_metrics = []
            for seed, sg in g.groupby('train_seed'):
                m = per_seed_metrics(sg['y_true'], sg['prob'])
                if m is not None:
                    seed_metrics.append(m)
            agg = {}
            for k in METRICS:
                vals = [m[k] for m in seed_metrics]
                agg[k] = np.mean(vals)
                agg[f'{k}_std'] = np.std(vals)
            results[(setting, method)] = agg
            rows.append({'setting': setting, 'method': method,
                         'n_seeds': len(seed_metrics),
                         'auroc_mean': agg['auroc'], 'auroc_std': agg['auroc_std'],
                         'f1_mean': agg['f1'], 'f1_std': agg['f1_std'],
                         'acc_mean': agg['acc'], 'acc_std': agg['acc_std'],
                         'aupr_mean': agg['aupr'], 'aupr_std': agg['aupr_std']})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)
    print(f'\nSaved summary -> {OUT_CSV}')

    for k, label in [('auroc', 'AUROC'), ('f1', 'F1'), ('acc', 'ACC'), ('aupr', 'AUPR')]:
        print(f'\n===== {label} (5-seed mean±std) =====')
        print(f'{"method":<22} {"common":<16} {"fewer":<16} {"rare":<16}')
        for method in METHOD_ORDER:
            cells = []
            for setting in SETTING_ORDER:
                r = results[(setting, method)]
                cells.append(f'{r[k]:.4f}±{r[k+"_std"]:.4f}')
            print(f'{method:<22} {cells[0]:<16} {cells[1]:<16} {cells[2]:<16}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x = range(len(SETTING_ORDER))

    for ax, k in zip(axes.flat, METRICS):
        for method in METHOD_ORDER:
            vals = [results[(s, method)][k] for s in SETTING_ORDER]
            errs = [results[(s, method)][f'{k}_std'] for s in SETTING_ORDER]
            ax.errorbar(x, vals, yerr=errs, label=method, color=COLORS[method],
                        marker=MARKERS[method], linestyle=LINESTYLES[method],
                        linewidth=2, markersize=8, capsize=3, markeredgewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER], fontsize=9)
        ax.set_ylabel(k.upper(), fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0, 0].legend(fontsize=8.5, loc='upper right', framealpha=0.9, ncol=1)
    fig.suptitle('EviDDIE Ablation Study — Zero-Shot Setting (5 seeds)',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'\nSaved figure -> {OUT_PNG}')


if __name__ == '__main__':
    main()
