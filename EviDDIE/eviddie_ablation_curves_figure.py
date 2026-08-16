#!/usr/bin/env python
# coding=utf-8
"""EviDDIE 消融训练曲线图 (2026-08-16)：dev AUROC/F1/loss 随训练迭代的
5-seed 均值折线 ± 标准差带。3 个变体头（softmax / w/o EVI / w/o BSA），
frozen 骨干，仅验证集口径。"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
PREFIXES = ['eviddie_new_s1', 'eviddie_new_s2', 'eviddie_new_s3',
            'eviddie_new_s4', 'eviddie_new_s5']
VARIANTS = ['softmax', 'evi_no_evi', 'wo_BSA', 'evi_full']
VARIANT_LABELS = {'softmax': 'Softmax baseline', 'evi_no_evi': 'EviDDIE w/o EVI',
                  'wo_BSA': 'EviDDIE w/o BSA', 'evi_full': 'EviDDIE (EDL head retrained)'}
COLORS = {'softmax': '#999999', 'evi_no_evi': '#2196F3', 'wo_BSA': '#FF9800',
          'evi_full': '#7B1FA2'}   # 紫色（EDL 头重训）
COLORS_full = '#D32F2F'      # 生产 checkpoint 参考线（红）
LINESTYLE_FULL = (0, (4, 2))  # 长虚线
OUT_PNG = 'EviDDIE_Ablation_Curves.png'


def load_curves():
    frames = []
    for seed, prefix in zip(SEEDS, PREFIXES):
        p = f'results/ablation_curves_{prefix}_seed{seed}.csv'
        d = pd.read_csv(p)
        d['train_seed'] = seed
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_curves()
    print(f'Loaded {len(df)} rows: {sorted(df.variant.unique())} '
          f'x seeds {sorted(df.train_seed.unique())}')

    # 每 (variant, iter) 聚合 5 种子 mean±std
    agg = df.groupby(['variant', 'iter']).agg(
        au_mean=('dev_auroc', 'mean'), au_std=('dev_auroc', 'std'),
        f1_mean=('dev_f1', 'mean'), f1_std=('dev_f1', 'std'),
        loss_mean=('train_loss', 'mean'),
    ).reset_index()

    # 只画 AUROC 和 F1：不同变体的损失函数不可比（CE vs MSE vs EDL）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [('au', 'dev_auroc', 'Dev AUROC'),
              ('f1', 'dev_f1', 'Dev F1')]
    for ax, (key, _, title) in zip(axes, panels):
        for v in VARIANTS:
            g = agg[agg.variant == v].sort_values('iter')
            y = g[f'{key}_mean']
            sd = g[f'{key}_std']
            ax.plot(g['iter'], y, label=VARIANT_LABELS[v],
                    color=COLORS[v], linewidth=2)
            ax.fill_between(g['iter'], y - sd, y + sd,
                            color=COLORS[v], alpha=0.15)

        # 生产 checkpoint（骨干+头联合训练的完整模型）：冻结不重训 →
        # 水平参考线 + 标准差带（同一内部 dev 口径）
        if key in ('au', 'f1'):
            full = pd.read_csv('results/full_evi_dev_internal.csv')
            col = 'dev_auroc' if key == 'au' else 'dev_f1'
            fm, fs = full[col].mean(), full[col].std()
            ax.axhline(fm, color=COLORS_full, linestyle=LINESTYLE_FULL,
                       linewidth=2, label='EviDDIE (production checkpoint)')
            ax.axhspan(fm - fs, fm + fs, color=COLORS_full, alpha=0.12)
            ax.text(100, fm + fs + 0.012, f'{fm:.3f}±{fs:.3f}',
                    color=COLORS_full, fontsize=9, fontweight='bold',
                    ha='left', va='bottom')
        ax.set_xlabel('Training iteration', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    fig.suptitle('EviDDIE Frozen-Backbone Ablation — Training Curves (dev, 5 seeds)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {OUT_PNG}')

    # 终值摘要（step=5000 处的 dev 指标，5种子 mean±std）
    print('\n===== Final dev (step=5000, 5-seed mean±std) =====')
    final = agg[agg.iter == 5000]
    for v in VARIANTS:
        r = final[final.variant == v].iloc[0]
        print(f'{VARIANT_LABELS[v]:<30} AUROC={r["au_mean"]:.4f}±{r["au_std"]:.4f} '
              f'F1={r["f1_mean"]:.4f}±{r["f1_std"]:.4f}')


if __name__ == '__main__':
    main()
