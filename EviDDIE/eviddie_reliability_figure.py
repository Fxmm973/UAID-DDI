#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'results/predictions/predictions_eviddie_new_ablation.csv'
OUT = 'reliability_diagram_new.png'

N_BINS = 10
SETTINGS = ['fewer', 'rare']
TITLES = {'fewer': 'Fewer (test)', 'rare': 'Rare (test2)'}
COL_NATIVE = '#D32F2F'
COL_TSCALE = '#2196F3'


def fit_temperature(df):
    temps = {}
    dev = df[(df.setting == 'common') & (df.method == 'EviDDIE')]
    for seed, g in dev.groupby('train_seed'):
        p = np.clip(g['prob'].values, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        y = g['y_true'].values
        best_t, best_nll = 1.0, np.inf
        for t in np.arange(0.5, 5.01, 0.01):
            q = 1 / (1 + np.exp(-logit / t))
            qc = np.clip(q, 1e-6, 1 - 1e-6)
            nll = -np.mean(y * np.log(qc) + (1 - y) * np.log(1 - qc))
            if nll < best_nll:
                best_nll, best_t = nll, t
        temps[seed] = best_t
    return temps


def reliability(probs, y, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(probs, edges[1:-1])
    centers, freq, counts = [], [], []
    for b in range(n_bins):
        m = ids == b
        counts.append(int(m.sum()))
        if m.sum() > 0:
            centers.append((edges[b] + edges[b + 1]) / 2)
            freq.append(y[m].mean())
        else:
            centers.append((edges[b] + edges[b + 1]) / 2)
            freq.append(np.nan)
    freq = np.array(freq)
    return np.array(centers), freq, np.array(counts), len(probs)


def per_seed_ece_mean(df, setting, method, n_bins=N_BINS):
    vals = []
    g = df[(df.setting == setting) & (df.method == method)]
    for seed, sg in g.groupby('train_seed'):
        p = sg['prob'].values
        y = sg['y_true'].values
        edges = np.linspace(0, 1, n_bins + 1)
        ids = np.digitize(p, edges[1:-1])
        ece = 0.0
        for b in range(n_bins):
            m = ids == b
            if m.sum() == 0:
                continue
            ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
        vals.append(ece)
    return float(np.mean(vals))


def main():
    df = pd.read_csv(CSV)
    native = df[df.method == 'EviDDIE']
    temps = fit_temperature(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, setting in zip(axes, SETTINGS):
        g = native[native.setting == setting]
        probs = g['prob'].values
        y = g['y_true'].values
        centers, freq, counts, n = reliability(probs, y)
        ece = per_seed_ece_mean(df, setting, 'EviDDIE')
        ax.bar(centers, freq, width=0.075, alpha=0.85, color=COL_NATIVE,
               label=f'EviDDIE (ECE={ece:.4f})', edgecolor='none')

        ts_probs = []
        for seed, sg in g.groupby('train_seed'):
            p = np.clip(sg['prob'].values, 1e-6, 1 - 1e-6)
            logit = np.log(p / (1 - p))
            ts_probs.append(1 / (1 + np.exp(-logit / temps[seed])))
        ts_probs = np.concatenate(ts_probs)
        tc, tf, tcounts, _ = reliability(ts_probs, y)
        ts_ece_vals = []
        for seed, sg in g.groupby('train_seed'):
            p = np.clip(sg['prob'].values, 1e-6, 1 - 1e-6)
            logit = np.log(p / (1 - p))
            tp = 1 / (1 + np.exp(-logit / temps[seed]))
            edges = np.linspace(0, 1, N_BINS + 1)
            ids = np.digitize(tp, edges[1:-1])
            e = 0.0
            for b in range(N_BINS):
                m = ids == b
                if m.sum() == 0:
                    continue
                e += (m.sum() / len(tp)) * abs(sg['y_true'].values[m].mean() - tp[m].mean())
            ts_ece_vals.append(e)
        tece = float(np.mean(ts_ece_vals))
        ax.plot(tc, tf, color=COL_TSCALE, marker='o', markersize=5,
                linewidth=2, label=f'EviDDIE + TempScale (ECE={tece:.4f})')

        for cx, cnt in zip(centers, counts):
            if cnt > 0:
                ax.text(cx, 0.02, str(cnt), ha='center', va='bottom',
                        fontsize=7, color='gray')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        ax.set_xlabel('Predicted probability', fontsize=10)
        ax.set_ylabel('Observed positive fraction', fontsize=10)
        ax.set_title(f'{TITLES[setting]}  (n={n})', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

    fig.suptitle('Reliability Diagrams — EviDDIE Zero-Shot (5-seed pooled)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {OUT}')
    print(f'Temperatures per seed: {temps}')


if __name__ == '__main__':
    main()
