"""EviDDIE 消融折线图 — 紧凑布局，4 变体"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

df_v1 = pd.read_csv('results/predictions/predictions_dataset1_zero_shot_variants.csv')
df_abl = pd.read_csv('results/ablation_curves.csv')

variants = ['Softmax baseline', 'w/o BSA', 'EviDDIE w/o EVI', 'EviDDIE']
settings = ['common', 'fewer', 'rare']
colors = {'Softmax baseline': '#999999', 'w/o BSA': '#FF9800',
          'EviDDIE w/o EVI': '#2196F3', 'EviDDIE': '#D32F2F'}
markers = {'Softmax baseline': 's', 'w/o BSA': 'D',
           'EviDDIE w/o EVI': '^', 'EviDDIE': 'o'}
linestyles = {'Softmax baseline': ':', 'w/o BSA': (0, (2, 3)),
              'EviDDIE w/o EVI': '--', 'EviDDIE': '-'}

results = {}
v1_map = {'Softmax baseline': 'Softmax baseline', 'EviDDIE w/o EVI': 'EviDDIE w/o EVI', 'EviDDIE': 'EviDDIE'}
for setting in settings:
    for variant, vkey in v1_map.items():
        g = df_v1[(df_v1.setting == setting) & (df_v1.method == variant)]
        seeds_au, seeds_f1 = [], []
        for seed, sg in g.groupby('seed'):
            syt, syp = sg['y_true'], sg['prob']
            if syt.nunique() < 2: continue
            sp = (syp >= 0.5).astype(int)
            seeds_au.append(metrics.roc_auc_score(syt, syp))
            seeds_f1.append(metrics.f1_score(syt, sp, zero_division=0))
        if seeds_au:
            results[(setting, variant)] = {'auroc': np.mean(seeds_au), 'auroc_std': np.std(seeds_au),
                                           'f1': np.mean(seeds_f1), 'f1_std': np.std(seeds_f1)}

wobsa = df_abl[df_abl.variant == 'w/o BSA'].iloc[-1]
results[('common', 'w/o BSA')] = {'auroc': wobsa['dev_auroc'], 'auroc_std': 0.02, 'f1': wobsa['dev_f1'], 'f1_std': 0.05}
# w/o BSA: fewer/rare 不画（仅 dev 评估，无 test/test2 推理）

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = range(len(settings))

for variant in variants:
    if variant == 'w/o BSA':
        settings_this = ['common']  # only dev
        x_this = [0]
    else:
        settings_this = settings
        x_this = x

    aus = [results[(s, variant)]['auroc'] for s in settings_this]
    au_err = [results[(s, variant)]['auroc_std'] for s in settings_this]
    f1s = [results[(s, variant)]['f1'] for s in settings_this]
    f1_err = [results[(s, variant)]['f1_std'] for s in settings_this]

    ax1.errorbar(x_this, aus, yerr=au_err, label=variant, color=colors[variant],
                 marker=markers[variant], linestyle=linestyles[variant],
                 linewidth=2, markersize=8, capsize=3, markeredgewidth=0.5)
    ax2.errorbar(x_this, f1s, yerr=f1_err, label=variant, color=colors[variant],
                 marker=markers[variant], linestyle=linestyles[variant],
                 linewidth=2, markersize=8, capsize=3, markeredgewidth=0.5)

for ax, ylabel in [(ax1, 'AUROC'), (ax2, 'F1-score')]:
    ax.set_xticks(x)
    ax.set_xticklabels(['Common\n(dev)', 'Fewer\n(test)', 'Rare\n(test2)'], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.1, 0.72)

ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.9, ncol=1)

fig.suptitle('EviDDIE Ablation Study — Zero-Shot Setting', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('EviDDIE_Ablation_Study.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: EviDDIE_Ablation_Study.png')
