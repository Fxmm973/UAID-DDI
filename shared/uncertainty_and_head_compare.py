#!/usr/bin/env python
# coding=utf-8
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist, ttest_rel
from sklearn.metrics import roc_auc_score


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def nll(p, y):
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def ece(p, y, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    ids = np.digitize(p, bins[1:-1])
    e = 0.0
    for b in range(nb):
        m = ids == b
        if m.sum():
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(e)


METRICS = {'brier': brier, 'nll': nll, 'ece': ece}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation-csv',
                    default='EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv')
    ap.add_argument('--frozen-csv',
                    default='EviDDIE/results/predictions/predictions_evi_full_frozen.csv')
    ap.add_argument('--out-dir', default='EviDDIE/results')
    args = ap.parse_args()

    ab = pd.read_csv(args.ablation_csv)
    fz = pd.read_csv(args.frozen_csv)

    # 1) 同骨干同 frozen 协议的 softmax vs EDL paired 对照（P0-6）
    paired = []
    for setting in ['fewer', 'rare']:
        a = ab[(ab['setting'] == setting) & (ab['method'] == 'Softmax baseline')]
        b = fz[fz['setting'] == setting]
        seeds = sorted(set(a['train_seed']))
        for metric, fn in METRICS.items():
            av = [fn(a[a['train_seed'] == s]['prob'].values,
                     a[a['train_seed'] == s]['y_true'].values) for s in seeds]
            bv = [fn(b[b['train_seed'] == s]['prob'].values,
                     b[b['train_seed'] == s]['y_true'].values) for s in seeds]
            t, p = ttest_rel(av, bv)
            diff = np.mean(av) - np.mean(bv)
            tv = tdist.ppf(0.975, len(seeds) - 1)
            se = np.std(np.array(av) - np.array(bv), ddof=1) / np.sqrt(len(seeds))
            paired.append({'setting': setting, 'metric': metric,
                           'softmax_mean': np.mean(av), 'edl_mean': np.mean(bv),
                           'diff': diff, 'ci95_low': diff - tv * se,
                           'ci95_high': diff + tv * se, 'p_paired_t': p,
                           'n_seeds': len(seeds)})
    pd.DataFrame(paired).to_csv(os.path.join(args.out_dir, 'head_paired.csv'), index=False)
    print(f"[saved] {os.path.join(args.out_dir, 'head_paired.csv')}")

    # 2) uncertainty 质量：error-detection AUROC 与 Spearman(u, error)（P0-6）
    quality = []
    for setting in ['fewer', 'rare']:
        for method in ['EviDDIE', 'Softmax baseline']:
            sub = ab[(ab['setting'] == setting) & (ab['method'] == method)]
            errs, uncs = [], []
            for s, g in sub.groupby('train_seed'):
                p = g['prob'].values
                y = g['y_true'].values
                u = g['uncertainty'].values if 'uncertainty' in g else np.maximum(p, 1 - p)
                errs.append(((p >= 0.5).astype(int) != y).astype(float))
                uncs.append(u)
            err = np.concatenate(errs)
            unc = np.concatenate(uncs)
            det_auc = roc_auc_score(err, unc) if len(np.unique(err)) > 1 else np.nan
            sp = spearmanr(unc, err).statistic
            quality.append({'setting': setting, 'method': method,
                            'uncertainty_source': 'u_EDL' if method == 'EviDDIE' else 'MSP',
                            'error_detection_auroc': det_auc,
                            'spearman_u_error': sp})
    pd.DataFrame(quality).to_csv(os.path.join(args.out_dir, 'uncertainty_quality.csv'),
                                 index=False)
    print(f"[saved] {os.path.join(args.out_dir, 'uncertainty_quality.csv')}")


if __name__ == '__main__':
    main()
