#!/usr/bin/env python
# coding=utf-8
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ['auroc', 'acc', 'f1_score']
SHOTS = [1, 5]
SETTING_LABELS = {'common': 'Common (dev)', 'test': 'Fewer (test)', 'test2': 'Rare (test2)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/validation/weight_sweep.csv')
    ap.add_argument('--out', default='results/validation/fig_weight_sweep.png')
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f'CSV not found: {args.csv}')

    df = pd.read_csv(args.csv)
    weights = sorted(df['weight'].unique())

    fig, axes = plt.subplots(len(SHOTS), 1, figsize=(8, 9), squeeze=False)
    for si, shot in enumerate(SHOTS):
        ax = axes[si][0]
        sub = df[df['shot'] == shot]
        for m in METRICS:
            for setting in sub['setting'].dropna().unique():
                s = sub[(sub['setting'] == setting) & (sub['metric'] == m)]
                ax.plot(s['weight'], s['value'], marker='o',
                        label=f'{m} ({SETTING_LABELS.get(setting, setting)})')
        ax.set_xlabel('SHCR proxy weight')
        ax.set_ylabel('Metric value')
        ax.set_title(f'{shot}-shot supervision')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle('Sensitivity of PharDDIE to the SHCR proxy weight (archived records)')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
