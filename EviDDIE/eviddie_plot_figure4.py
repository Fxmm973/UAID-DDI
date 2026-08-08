#!/usr/bin/env python
# coding=utf-8
"""EviDDIE 消融验证曲线 — Figure 4 (v2: 4 variants)"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})

VARIANT_STYLES = {
    'softmax':   ('Softmax baseline',      '#AAAAAA', ':',  1.5),
    'w/o BSA':   ('EviDDIE w/o BSA',        '#FF9800', '--', 1.8),
    'evi_no_evi':('EviDDIE w/o EVI',        '#2196F3', '-.', 1.8),
    'full_evi':  ('EviDDIE (full)',          '#D32F2F', '-',  2.2),
}

df = pd.read_csv('results/ablation_curves.csv')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for variant, (label, color, ls, lw) in VARIANT_STYLES.items():
    vdf = df[df['variant'] == variant].sort_values('iter')
    axes[0].plot(vdf['iter'], vdf['dev_auroc'], label=label,
                 color=color, linestyle=ls, linewidth=lw)
    axes[1].plot(vdf['iter'], vdf['dev_f1'], label=label,
                 color=color, linestyle=ls, linewidth=lw)
    axes[2].plot(vdf['iter'], vdf['dev_acc'], label=label,
                 color=color, linestyle=ls, linewidth=lw)

for ax in axes:
    ax.set_xlabel('Training iterations', fontsize=11)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('AUROC', fontsize=12)
axes[0].set_title('Dev Set AUROC', fontsize=13, fontweight='bold')
axes[1].set_ylabel('F1-score', fontsize=12)
axes[1].set_title('Dev Set F1-score', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Accuracy', fontsize=12)
axes[2].set_title('Dev Set Accuracy', fontsize=13, fontweight='bold')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, frameon=True)

fig.suptitle('EviDDIE Ablation Study — Validation Set Curves',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig('EviDDIE_Ablation_Study.png', dpi=200, bbox_inches='tight')
plt.savefig('EviDDIE_Ablation_Study.pdf', bbox_inches='tight')
print('Saved: EviDDIE_Ablation_Study.png / .pdf')
