#!/usr/bin/env python
# coding=utf-8
"""
P0-3 residual fix (2026-08-16): independent Random row for the triage
prioritization table (tab:triage_prioritization).

The legacy Random row mixed two sources: P@K came from a random retained set
while referral/coverage were read from the UA mask. Per the P0-3 protocol the
Random strategy must be independent end-to-end: draw a random retained set of
the SAME coverage as the UA operating point, 200 times per seed, and report
P@K (mean +- SD) and selective risk (mean +- 95% CI).

Usage:
  python shared/rq3_triage_priority_random.py --csv PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv
"""
import argparse
import numpy as np
import pandas as pd

# UA operating-point automatic coverages from the paper's triage table
UA_COVERAGE = {'fewer': 0.4152, 'rare': 0.4224}
N_REPS = 200


def random_row(g, coverage, rng):
    """One independent random draw: keep `coverage` fraction, report P@K and risk."""
    n = len(g)
    k_keep = max(1, int(round(n * coverage)))
    keep = rng.permutation(n)[:k_keep]
    kept = g.iloc[keep]
    # candidate-priority ordering within the retained set, as in the paper's P@K
    top = kept.sort_values('prob', ascending=False)
    p_at = {}
    for K in (10, 20, 50):
        t = top.head(K)
        p_at[K] = t['y_true'].mean() if len(t) >= K else np.nan
    risk = 1.0 - kept['y_true'].mean()
    return p_at, risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--seeds', nargs='+', type=int,
                    default=[19940419, 20230801, 20240115, 20240520, 20240910])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'train_seed' not in df.columns and 'seed' in df.columns:
        df = df.rename(columns={'seed': 'train_seed'})

    for setting in ['fewer', 'rare']:
        sub = df[(df['setting'] == setting) & (df['shot'] == 1)]
        cov = UA_COVERAGE[setting]
        p10s, p20s, p50s, risks = [], [], [], []
        for s in args.seeds:
            g = sub[sub['train_seed'] == s]
            if g.empty:
                continue
            rng = np.random.default_rng(int(s) * 100003 + 42)
            reps = [random_row(g, cov, rng) for _ in range(N_REPS)]
            p10s.append(np.nanmean([r[0][10] for r in reps]))
            p20s.append(np.nanmean([r[0][20] for r in reps]))
            p50s.append(np.nanmean([r[0][50] for r in reps]))
            risks.append(np.nanmean([r[1] for r in reps]))
        p10, p20, p50 = np.mean(p10s), np.mean(p20s), np.mean(p50s)
        risk = np.mean(risks)
        sd10, sd20, sd50 = np.std(p10s, ddof=1), np.std(p20s, ddof=1), np.std(p50s, ddof=1)
        risk_sd = np.std(risks, ddof=1)
        # per-seed mean over reps has tiny sampling noise; CI across seeds via t_0.975,4
        tcrit = 2.776
        risk_ci = tcrit * risk_sd / np.sqrt(len(risks))
        print(f'{setting.capitalize()} (1-shot) | Random | P@10 {p10:.4f}±{sd10:.4f} | '
              f'P@20 {p20:.4f}±{sd20:.4f} | P@50 {p50:.4f}±{sd50:.4f} | '
              f'referral {1-cov:.4f} | coverage {cov:.4f} | risk {risk:.4f}±{risk_ci:.4f}')


if __name__ == '__main__':
    main()
