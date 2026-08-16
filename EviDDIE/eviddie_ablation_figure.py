#!/usr/bin/env python
# coding=utf-8
"""EviDDIE 消融主图 (2026-08-16)：AUROC + F1 两面板，3 settings × 4 variants，
5-seed mean±std，F1 面板标注配对 t 检验显著性 (*p<0.10, **p<0.05 vs EviDDIE)。"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats

CSV_PATH = 'results/predictions/predictions_eviddie_new_ablation.csv'
OUT_MAIN = 'EviDDIE_Ablation_Study.png'          # 正文主图（覆盖旧图）
OUT_SUPP = 'EviDDIE_Ablation_Study_4metrics.png'  # 补充材料（4 指标）

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
VARIANT_ABL = ['Softmax baseline', 'EviDDIE w/o BSA', 'EviDDIE w/o EVI']


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


def build_results(df):
    results = {}
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
            agg['seed_vals'] = seed_metrics
            results[(setting, method)] = agg
    return results


def sig_stars(df, setting, metric):
    """配对比 t 检验：消融变体 vs EviDDIE。返回 {variant: (star, y_offset)}"""
    stars = {}
    full_g = df[(df.setting == setting) & (df.method == 'EviDDIE')]
    full_vals = []
    for seed, sg in full_g.groupby('train_seed'):
        m = per_seed_metrics(sg['y_true'], sg['prob'])
        if m is not None:
            full_vals.append(m[metric])
    for v in VARIANT_ABL:
        vg = df[(df.setting == setting) & (df.method == v)]
        v_vals = []
        for seed, sg in vg.groupby('train_seed'):
            m = per_seed_metrics(sg['y_true'], sg['prob'])
            if m is not None:
                v_vals.append(m[metric])
        t, p = stats.ttest_rel(v_vals, full_vals)
        stars[v] = '**' if p < 0.05 else ('*' if p < 0.10 else '')
    return stars


def main():
    df = pd.read_csv(CSV_PATH)
    results = build_results(df)
    x = range(len(SETTING_ORDER))

    # ---- 正文主图：AUROC + F1 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for method in METHOD_ORDER:
        aus = [results[(s, method)]['auroc'] for s in SETTING_ORDER]
        au_err = [results[(s, method)]['auroc_std'] for s in SETTING_ORDER]
        f1s = [results[(s, method)]['f1'] for s in SETTING_ORDER]
        f1_err = [results[(s, method)]['f1_std'] for s in SETTING_ORDER]
        ax1.errorbar(x, aus, yerr=au_err, label=method, color=COLORS[method],
                     marker=MARKERS[method], linestyle=LINESTYLES[method],
                     linewidth=2, markersize=8, capsize=3, markeredgewidth=0.5)
        ax2.errorbar(x, f1s, yerr=f1_err, label=method, color=COLORS[method],
                     marker=MARKERS[method], linestyle=LINESTYLES[method],
                     linewidth=2, markersize=8, capsize=3, markeredgewidth=0.5)

    # F1 面板：显著性星号（消融变体 vs EviDDIE）
    for si, setting in enumerate(SETTING_ORDER):
        stars = sig_stars(df, setting, 'f1')
        for vi, v in enumerate(VARIANT_ABL):
            if stars[v]:
                f1_full = results[(setting, 'EviDDIE')]['f1']
                ax2.text(si - 0.18 + vi * 0.12, f1_full + 0.055, stars[v],
                         fontsize=11, color='black', fontweight='bold', ha='center')

    for ax, ylabel in [(ax1, 'AUROC'), (ax2, 'F1-score')]:
        ax.set_xticks(list(x))
        ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.1, 0.72)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.9, ncol=1)
    fig.suptitle('EviDDIE Ablation Study — Zero-Shot Setting (5 seeds)',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_MAIN, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved main figure -> {OUT_MAIN}')

    # ---- 补充材料：4 指标 2×2 ----
    fig2, axes = plt.subplots(2, 2, figsize=(13, 10))
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
    fig2.suptitle('EviDDIE Ablation Study — Zero-Shot Setting (5 seeds, 4 metrics)',
                  fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_SUPP, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved supplement figure -> {OUT_SUPP}')

    # ---- 表格：正文用（F1 显著，其余 ns）----
    print('\n===== 正文消融表（4指标 mean±std，F1显著性）=====')
    print(f'{"method":<22}{"setting":<9}{"AUROC":<18}{"F1":<20}{"ACC":<18}{"AUPR":<18}')
    for setting in SETTING_ORDER:
        for method in METHOD_ORDER:
            r = results[(setting, method)]
            star = ''
            if method != 'EviDDIE':
                stars = sig_stars(df, setting, 'f1')
                star = stars[method]
            print(f'{method:<22}{setting:<9}'
                  f'{r["auroc"]:.3f}±{r["auroc_std"]:.3f}{"":<8}'
                  f'{r["f1"]:.3f}±{r["f1_std"]:.3f}{star:<3}'
                  f'{r["acc"]:.3f}±{r["acc_std"]:.3f}{"":<8}'
                  f'{r["aupr"]:.3f}±{r["aupr_std"]:.3f}')


if __name__ == '__main__':
    main()
