#!/usr/bin/env python
# coding=utf-8
"""
RQ3 selective-referral evaluation, rebuilt to P0-3 reviewer protocol (2026-08-16).

Differences from the legacy pharddie_table4_paper.py:
  * Signals are separate strategies, each keeping/referring candidates by its OWN
    ranking: raw positive score p (candidate-priority semantics), MSP = max(p,1-p),
    margin = |p-0.5|, entropy H(p), the model's native u column (latent dispersion
    for PharDDIE >=5-shot, u_EDL for EviDDIE zero-shot), and TRUE random referral.
  * MSP / margin / entropy rank-equivalence for binary classification is verified
    numerically (Spearman rho) and printed, supporting the paper claim that the
    1-shot triage is a confidence-based reject option.
  * The Random baseline is independent: for every (seed, coverage) it draws
    referral sets R times and reports mean +- 95% CI.
  * Risk-coverage curves and AURC are computed PER SEED first, then aggregated
    as mean +- SD (never pooled).
  * Error-detection AUROC/AUPRC is reported for the native u signal and for the
    confidence (MSP) signal, giving the incremental value of u beyond confidence.

Usage:
  python shared/rq3_selective_referral.py --csv <predictions.csv> \
      --out results/rq3_<name>.csv [--settings fewer rare] [--shots 0 1 5] \
      [--n_random 200] [--coverage 0.30 0.50 0.70]
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

GRID = np.arange(0.05, 1.0, 0.05)  # risk-coverage curve grid


def load_predictions(csv):
    df = pd.read_csv(csv)
    # accept both legacy (v1) and v2 column names for the seed
    if 'train_seed' not in df.columns and 'seed' in df.columns:
        df = df.rename(columns={'seed': 'train_seed'})
    assert {'train_seed', 'setting', 'shot', 'y_true', 'prob'}.issubset(df.columns), \
        f'missing required columns: {list(df.columns)}'
    df['prob'] = df['prob'].astype(float)
    df['y_true'] = df['y_true'].astype(int)
    if 'uncertainty' not in df.columns:
        df['uncertainty'] = np.nan
    else:
        df['uncertainty'] = pd.to_numeric(df['uncertainty'], errors='coerce')
    p = df['prob'].clip(1e-12, 1 - 1e-12)
    df['p'] = df['prob']  # signal name 'p' = raw positive score column
    df['u'] = df['uncertainty']  # signal name 'u' = model-native uncertainty column
    df['msp'] = np.maximum(p, 1 - p)
    df['margin'] = np.abs(p - 0.5)
    df['entropy'] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return df


def verify_rank_equivalence(df):
    """MSP/margin/entropy must induce the same ranking for binary p (P0-3 claim)."""
    p = df['prob'].clip(1e-12, 1 - 1e-12).values
    msp, margin = np.maximum(p, 1 - p), np.abs(p - 0.5)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    r1 = spearmanr(msp, margin).correlation
    r2 = spearmanr(msp, ent).correlation
    r3 = spearmanr(margin, ent).correlation
    # msp/margin are monotonically INCREASING in confidence, entropy DECREASING,
    # so the expected signature is +1 / -1 / -1 (all |rho| = 1).
    print(f'[rank-equivalence] rho(msp,margin)={r1:.8f} rho(msp,entropy)={r2:.8f} '
          f'rho(margin,entropy)={r3:.8f} (expected +1 / -1 / -1)')
    return max(abs(r1), abs(r2), abs(r3))


def det_seed(train_seed, signal, setting, shot, coverage=None):
    """Deterministic rng seed (no builtin hash(): unstable across processes)."""
    base = int(train_seed) * 1000003
    sig_idx = {'p': 1, 'msp': 2, 'margin': 3, 'entropy': 4, 'u': 5, 'random': 6}[signal]
    cov = int(round(coverage * 1000)) if coverage is not None else 0
    return (base + sig_idx * 7919 + cov * 31) % (2 ** 32)


def keep_mask_from_signal(sig_values, keep_frac, signal, rng=None):
    """Return boolean mask of the KEPT (automatically processed) candidates.
    signal semantics: 'p' keeps highest p (candidate priority, legacy probability-only);
    'msp'/'margin' keep the most confident; 'entropy'/'u' keep the least uncertain;
    'random' keeps a uniform random subset."""
    n = len(sig_values)
    k = max(1, int(round(n * keep_frac)))
    if signal == 'random':
        idx = rng.permutation(n)[:k]
    elif signal in ('entropy', 'u'):
        idx = np.argsort(np.nan_to_num(sig_values, nan=np.inf))[:k]
    else:  # p / msp / margin: keep the highest values
        idx = np.argsort(sig_values)[::-1][:k]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def selective_risk(y_true, keep_mask):
    kept = y_true[keep_mask]
    if len(kept) == 0:
        return np.nan
    return 1.0 - kept.mean()  # error rate on the retained (automatic) set


def per_seed_curve(y_true, sig_values, signal, rng=None, n_random=200):
    """Risk-coverage curve on the GRID for one seed. For 'random', the expected
    selective risk at every coverage is the population error rate (analytic), so
    the curve is constant."""
    if signal == 'random':
        return np.full(len(GRID), 1.0 - y_true.mean())
    risks = []
    for c in GRID:
        r = selective_risk(y_true, keep_mask_from_signal(sig_values, c, signal, rng))
        risks.append(r)
    return np.asarray(risks)


def aurc(risks):
    """Area under the risk-coverage curve (trapezoid; lower is better)."""
    return np.trapz(risks, GRID)


def random_referral_stats(y_true, coverage, rng, n_random=200):
    """True random referral at a fixed coverage: mean +- 95% CI over repeats."""
    rs = np.asarray([selective_risk(y_true, keep_mask_from_signal(
        y_true.astype(float), coverage, 'random', rng)) for _ in range(n_random)])
    m, s = rs.mean(), rs.std(ddof=1)
    ci = 1.96 * s / np.sqrt(len(rs))
    return m, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='results/rq3_selective_referral.csv')
    ap.add_argument('--settings', nargs='+', default=['fewer', 'rare'])
    ap.add_argument('--shots', nargs='+', type=int, default=[0, 1, 5])
    ap.add_argument('--n_random', type=int, default=200)
    ap.add_argument('--coverages', nargs='+', type=float, default=[0.30, 0.50, 0.70])
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='restrict to these training seeds (default: all in CSV)')
    args = ap.parse_args()

    df = load_predictions(args.csv)
    seeds = args.seeds or sorted(df['train_seed'].unique())
    seeds = [s for s in seeds if s in set(df['train_seed'].unique())]
    print(f'[seeds] {seeds}')

    rows = []
    for setting in args.settings:
        sub = df[df['setting'] == setting]
        if sub.empty:
            continue
        for shot in args.shots:
            ssub = sub[sub['shot'] == shot]
            if ssub.empty:
                continue
            print(f'[block] setting={setting} shot={shot} n={len(ssub)}')
            if shot <= 1:
                verify_rank_equivalence(ssub)

            # ---- per-seed AURC + curves (aggregated as mean +- SD) ----
            for signal in ['p', 'msp', 'margin', 'entropy', 'u', 'random']:
                per_seed_risks, per_seed_aurcs = [], []
                for s in seeds:
                    sd = ssub[ssub['train_seed'] == s]
                    if sd.empty:
                        continue
                    rng = np.random.default_rng(det_seed(s, signal, setting, shot))
                    y = sd['y_true'].values
                    sv = sd[signal].values if signal != 'random' else sd['prob'].values
                    if signal == 'u' and np.isnan(sv).all():
                        continue
                    curve = per_seed_curve(y, sv, signal, rng=rng, n_random=args.n_random)
                    per_seed_risks.append(curve)
                    per_seed_aurcs.append(aurc(curve))
                if not per_seed_aurcs:
                    continue
                risks = np.vstack(per_seed_risks)
                rows.append({'setting': setting, 'shot': shot, 'signal': signal,
                             'aurc_mean': float(np.mean(per_seed_aurcs)),
                             'aurc_sd': float(np.std(per_seed_aurcs, ddof=1)),
                             'curve_mean': ','.join(f'{v:.4f}' for v in risks.mean(0)),
                             'curve_sd': ','.join(f'{v:.4f}' for v in risks.std(0, ddof=1))})

            # ---- matched-coverage selective risk (fixed coverages) ----
            for c in args.coverages:
                for signal in ['p', 'msp', 'margin', 'entropy', 'u', 'random']:
                    vals = []
                    for s in seeds:
                        sd = ssub[ssub['train_seed'] == s]
                        if sd.empty:
                            continue
                        rng = np.random.default_rng(det_seed(s, signal, setting, shot, c))
                        y = sd['y_true'].values
                        sv = sd[signal].values if signal != 'random' else sd['prob'].values
                        if signal == 'u' and np.isnan(sv).all():
                            continue
                        if signal == 'random':
                            m, ci = random_referral_stats(y, c, rng, n_random=args.n_random)
                            vals.append((m, ci))
                        else:
                            mask = keep_mask_from_signal(sv, c, signal, rng=rng)
                            vals.append((selective_risk(y, mask), np.nan))
                    if vals:
                        means = [v[0] for v in vals]
                        cis = [v[1] for v in vals if not np.isnan(v[1])]
                        rows.append({'setting': setting, 'shot': shot, 'signal': signal,
                                     f'coverage_{c:.2f}_risk_mean': float(np.mean(means)),
                                     f'coverage_{c:.2f}_risk_sd': float(np.std(means, ddof=1)),
                                     f'coverage_{c:.2f}_random_ci_half': float(np.mean(cis)) if cis else np.nan})

            # ---- error-detection increment of u over confidence ----
            ed_rows = []
            for s in seeds:
                sd = ssub[ssub['train_seed'] == s]
                if sd.empty:
                    continue
                y = sd['y_true'].values
                pred = (sd['prob'].values >= 0.5).astype(int)
                err = (pred != y).astype(int)
                if err.sum() == 0 or (1 - err).sum() == 0:
                    continue
                # err_score oriented so that HIGHER = more likely an error;
                # msp (confidence) is negated, entropy/u are used as-is.
                for label, score, higher_is_uncertain in [('msp', sd['msp'].values, False),
                                                          ('entropy', sd['entropy'].values, True),
                                                          ('u', sd['uncertainty'].values, True)]:
                    sc = np.nan_to_num(score, nan=np.median(np.nan_to_num(score, nan=0)))
                    if np.allclose(sc, sc[0]):
                        continue
                    err_score = sc if higher_is_uncertain else -sc
                    auc = roc_auc_score(err, err_score)
                    pr = average_precision_score(err, err_score)
                    ed_rows.append({'setting': setting, 'shot': shot, 'seed': s,
                                    'signal': label, 'error_auroc': auc, 'error_auprc': pr})
            if ed_rows:
                ed = pd.DataFrame(ed_rows)
                for label in ['msp', 'entropy', 'u']:
                    e = ed[ed['signal'] == label]
                    if e.empty:
                        continue
                    rows.append({'setting': setting, 'shot': shot,
                                 'signal': f'err-detect-{label}',
                                 'error_auroc_mean': float(e['error_auroc'].mean()),
                                 'error_auroc_sd': float(e['error_auroc'].std(ddof=1)),
                                 'error_auprc_mean': float(e['error_auprc'].mean()),
                                 'error_auprc_sd': float(e['error_auprc'].std(ddof=1))})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f'[saved] {args.out} ({len(out)} rows)')


if __name__ == '__main__':
    main()
