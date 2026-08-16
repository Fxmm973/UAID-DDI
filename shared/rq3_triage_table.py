#!/usr/bin/env python
# coding=utf-8
"""
Rebuild tab:triage_prioritization (unified triage semantics, 1-shot).

Rows rebuilt under the P0-3/P0-4 protocol:
  Prob. only   : no referral; P@K and selective risk on the full retained set
  Prob. + unc. : unified four-action policy (tau_p=0.25, tau_u=0.35);
                 automatic set = {u_rank <= tau_u} (high-priority + low-priority),
                 referred set  = {u_rank >  tau_u} (expert referral + deferred review);
                 1-shot u = entropy H(p), percentile-ranked in [0,1] via the
                 validation (common) empirical CDF, matching the paper's
                 threshold-transfer semantics
  Random       : independent random referral at the SAME automatic coverage as
                 Prob.+unc. (200 repetitions per seed), P@K mean +- SD,
                 selective risk mean +- 95% CI

Output: table rows with P@10/P@20/P@50, referral rate, coverage, selective risk
(per-seed mean +- SD over the five training seeds).

Usage:
  python shared/rq3_triage_table.py --csv PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv
"""
import argparse
import numpy as np
import pandas as pd

TAU_P = 0.25
TAU_U = 0.35
N_REPS = 200
SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def p_at_k(top_p, top_y, K):
    t = pd.DataFrame({'p': top_p, 'y': top_y}).sort_values('p', ascending=False).head(K)
    return t['y'].mean() if len(t) >= K else np.nan


def err_rate(g):
    """Selective risk = classification error rate on the retained set (0.5 threshold)."""
    return float((g['y_pred'].values != g['y_true'].values).mean())


def prob_only_row(g):
    """No referral: full retained set."""
    return dict(p10=p_at_k(g['prob'], g['y_true'], 10),
                p20=p_at_k(g['prob'], g['y_true'], 20),
                p50=p_at_k(g['prob'], g['y_true'], 50),
                referral=0.0, coverage=1.0,
                risk=err_rate(g))


def ua_row(g, u_cdf):
    """Unified four-action policy; automatic set = {u_rank <= tau_u}."""
    u = entropy(g['prob'].values)
    u_rank = u_cdf(u)
    auto = u_rank <= TAU_U
    g_auto = g[auto]
    return dict(p10=p_at_k(g_auto['prob'], g_auto['y_true'], 10),
                p20=p_at_k(g_auto['prob'], g_auto['y_true'], 20),
                p50=p_at_k(g_auto['prob'], g_auto['y_true'], 50),
                referral=float(1.0 - auto.mean()), coverage=float(auto.mean()),
                risk=err_rate(g_auto))


def random_row(g, coverage, rng):
    """Independent random retained set at the UA automatic coverage."""
    n = len(g)
    k_keep = max(1, int(round(n * coverage)))
    keep = rng.permutation(n)[:k_keep]
    kept = g.iloc[keep]
    return dict(p10=p_at_k(kept['prob'], kept['y_true'], 10),
                p20=p_at_k(kept['prob'], kept['y_true'], 20),
                p50=p_at_k(kept['prob'], kept['y_true'], 50),
                referral=1.0 - coverage, coverage=coverage,
                risk=err_rate(kept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'train_seed' not in df.columns and 'seed' in df.columns:
        df = df.rename(columns={'seed': 'train_seed'})
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')
    df['y_true'] = df['y_true'].astype(int)

    for setting in ['fewer', 'rare']:
        sub = df[(df['setting'] == setting) & (df['shot'] == 1)]
        val = df[(df['setting'] == 'common') & (df['shot'] == 1)]
        # validation empirical CDF for entropy ranks
        val_u = entropy(val['prob'].values)
        val_u_sorted = np.sort(val_u)

        def u_cdf(u_vals):
            return np.searchsorted(val_u_sorted, u_vals) / len(val_u_sorted)

        print(f'=== {setting.capitalize()} (1-shot) ===')
        for strat, fn in [('Prob. only', prob_only_row), ('Prob. + unc.', ua_row)]:
            aggs = {k: [] for k in ['p10', 'p20', 'p50', 'referral', 'coverage', 'risk']}
            for s in SEEDS:
                g = sub[sub['train_seed'] == s]
                if g.empty:
                    continue
                r = fn(g, u_cdf) if strat == 'Prob. + unc.' else fn(g)
                for k in aggs:
                    aggs[k].append(r[k])
            means = {k: float(np.mean(v)) for k, v in aggs.items()}
            sds = {k: float(np.std(v, ddof=1)) for k, v in aggs.items()}
            print(f"  {strat:14s} | P@10 {means['p10']:.4f}±{sds['p10']:.4f} | "
                  f"P@20 {means['p20']:.4f}±{sds['p20']:.4f} | P@50 {means['p50']:.4f}±{sds['p50']:.4f} | "
                  f"referral {means['referral']:.4f} | coverage {means['coverage']:.4f} | "
                  f"risk {means['risk']:.4f}±{sds['risk']:.4f}")
        # Random at the UA coverage
        ua_cov = float(np.mean([ua_row(sub[sub['train_seed'] == s], u_cdf)['coverage']
                                for s in SEEDS if not sub[sub['train_seed'] == s].empty]))
        aggs = {k: [] for k in ['p10', 'p20', 'p50', 'risk']}
        for s in SEEDS:
            g = sub[sub['train_seed'] == s]
            if g.empty:
                continue
            rng = np.random.default_rng(int(s) * 100003 + 42)
            reps = [random_row(g, ua_cov, rng) for _ in range(N_REPS)]
            aggs['p10'].append(np.mean([r['p10'] for r in reps]))
            aggs['p20'].append(np.mean([r['p20'] for r in reps]))
            aggs['p50'].append(np.mean([r['p50'] for r in reps]))
            aggs['risk'].append(np.mean([r['risk'] for r in reps]))
        means = {k: float(np.mean(v)) for k, v in aggs.items()}
        sds = {k: float(np.std(v, ddof=1)) for k, v in aggs.items()}
        tcrit = 2.776  # t_0.975, df=4
        risk_ci = tcrit * sds['risk'] / np.sqrt(len(aggs['risk']))
        print(f"  {'Random':14s} | P@10 {means['p10']:.4f}±{sds['p10']:.4f} | "
              f"P@20 {means['p20']:.4f}±{sds['p20']:.4f} | P@50 {means['p50']:.4f}±{sds['p50']:.4f} | "
              f"referral {1-ua_cov:.4f} | coverage {ua_cov:.4f} | risk {means['risk']:.4f}±{risk_ci:.4f}")


if __name__ == '__main__':
    main()
