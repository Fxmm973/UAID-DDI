#!/usr/bin/env python
# coding=utf-8
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ABLATION_LABELS = {'no_meta(SHCR)': 'w/o SHCR', 'no_ACI': 'w/o ACI', 'no_SRAE': 'w/o SRAE'}
METRICS = ['auroc', 'acc', 'f1_score']
SHOTS = [1, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/validation/ablation_results.csv')
    ap.add_argument('--out', default='results/validation/fig_ablation.png')
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f'CSV not found: {args.csv}')

    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(len(SHOTS), 1, figsize=(8, 9), squeeze=False)
    for si, shot in enumerate(SHOTS):
        ax = axes[si][0]
        sub = df[(df['shot'] == shot) & (df['setting'] == 'TEST2')]
        order = ['no_meta(SHCR)', 'no_ACI', 'no_SRAE']
        xs = range(len(order))
        for xi, abname in enumerate(order):
            a = sub[sub['ablation'] == abname]
            for m, color in zip(METRICS, ['tab:blue', 'tab:orange', 'tab:green']):
                v = a[a['metric'] == m]['value'].values
                if len(v):
                    ax.bar(xi - 0.2 + METRICS.index(m) * 0.2, v[0], width=0.18,
                           color=color, alpha=0.85, label=m if xi == 0 else None)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([ABLATION_LABELS[a] for a in order])
        ax.set_ylabel('Metric value (rare/test2)')
        ax.set_title(f'{shot}-shot supervision (archived pre-unified records)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis='y')
    fig.suptitle('PharDDIE component ablation (archived records; see provenance.md)')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
