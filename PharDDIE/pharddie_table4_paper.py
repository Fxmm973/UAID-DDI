#!/usr/bin/env python
# coding=utf-8
"""
Table 4: Uncertainty-Aware Prioritization — unified triage semantics (reviewer P0-6).

Unified action grouping (matches the paper, Sec. Rule-Based Triage Policy):
    automatic set = {high-priority review (p>=tau_p, u<=tau_u),
                     low-priority assignment (otherwise)}
    referred set  = {expert referral (p>=tau_p, u>tau_u),
                     deferred review   (p< tau_p, u>tau_u)}
so that coverage = P(u <= tau_u) and referral = P(u > tau_u) everywhere,
including the matched-coverage section (single definition, no inconsistency).

Uncertainty signals (u):
    1-shot: u_entropy = H(p)   (confidence-derived baseline, NOT epistemic)
    5-shot: u_latent  = SRAE latent dispersion score
    (zero-shot EviDDIE u_EDL = 2/S is handled in the EviDDIE scripts)

Both signals are heavy-tailed, so we map them to percentile ranks in [0,1]
(robust to outliers). The test set is mapped through the VALIDATION empirical
CDF, so tau_u selected on validation transfers to the test set with a stable
meaning: tau_u = q keeps the (1-q) most uncertain candidates in the referred set.

Outputs (results/table4_paper.txt):
    - main table (Random / Probability only / Probability + uncertainty),
      with Random rows also reporting referral/coverage/selective risk
      under the same mask
    - per-setting tau_p / tau_u
    - threshold-sensitivity sweep (risk-coverage curve) + AURC
    - matched-coverage comparison (rare 1-shot)
    - fixed-referral-budget comparison
    - exact automatic / referred counts
"""
import pandas as pd
import numpy as np
import os

df = pd.read_csv('results/predictions/predictions_dataset1_PharDDIE.csv')
SEED_COL = 'train_seed' if 'train_seed' in df.columns else 'seed'
SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]


def find_thresholds(probs, uncs, labels):
    """tau_p: max F1; tau_u: min selective risk over automatic set with coverage >= 30%."""
    best_f1, best_tau_p = 0, 0.5
    for tau in np.arange(0.05, 1.0, 0.025):
        preds = (probs >= tau).astype(int)
        tp = (preds == 1) & (labels == 1)
        fp = (preds == 1) & (labels == 0)
        fn = (preds == 0) & (labels == 1)
        prec = tp.sum() / (tp.sum() + fp.sum()) if (tp.sum() + fp.sum()) > 0 else 0
        rec = tp.sum() / (tp.sum() + fn.sum()) if (tp.sum() + fn.sum()) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_tau_p = f1, tau

    # tau_u 网格基于分位数（uncs 已是 [0,1] 分位数秩）
    cand_taus = np.unique(np.quantile(uncs, np.linspace(0.05, 0.95, 31)))
    best_risk, best_tau_u = float('inf'), float(np.median(uncs))
    for tau in cand_taus:
        auto = uncs <= tau  # unified automatic set
        if auto.mean() < 0.30:
            continue
        risk = np.mean((probs[auto] >= best_tau_p).astype(int) != labels[auto]) if auto.sum() > 0 else 0
        if risk < best_risk:
            best_risk, best_tau_u = risk, tau
    return best_tau_p, best_tau_u


def compute_entropy(probs):
    """u_entropy = H(p), a confidence-derived baseline (NOT epistemic uncertainty)."""
    p = np.clip(probs, 1e-15, 1 - 1e-15)
    return (-p * np.log(p) - (1 - p) * np.log(1 - p)) / np.log(2)


def get_unc_raw(probs, uncs, shot):
    """Raw uncertainty signal: u_entropy (1-shot) or u_latent (K>=5)."""
    if shot == 1:
        return compute_entropy(probs)
    return uncs


def rank_scores(x):
    """Map scores to percentile ranks in [0,1] (robust to heavy tails)."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    return (np.argsort(np.argsort(x)) + 1.0) / n


def ecdf_map(raw, sorted_ref):
    """Map test scores to quantiles under the validation empirical CDF."""
    return np.clip(np.searchsorted(sorted_ref, raw, side='right') / len(sorted_ref), 0, 1)


def fmt(x):
    return '{:.4f}'.format(x)


def main():
    settings = ['fewer', 'rare']
    shots = [1]  # 论文只报告 1-shot 分诊结果；5/10-shot 不报告

    lines = []
    lines.append('=' * 120)
    lines.append('Table 4: Agent-Assisted Prioritization on Held-Out DDI Candidates')
    lines.append('Unified triage semantics (automatic = high-priority + low-priority;')
    lines.append('referred = expert referral + deferred review), so coverage = P(u <= tau_u).')
    lines.append('Uncertainty signal: u_entropy = H(p) for 1-shot;')
    lines.append('u_latent = SRAE latent dispersion score for 5-shot.')
    lines.append('Both signals are mapped to percentile ranks in [0,1] (robust to heavy tails);')
    lines.append('test scores are mapped through the validation empirical CDF so tau_u transfers.')
    lines.append('=' * 120)

    for setting in settings:
        s_label = 'Fewer-shot (test)' if setting == 'fewer' else 'Rare-event (test2)'
        lines.append('')
        lines.append('--- Setting: {} ---'.format(s_label))
        H = '{:<6} {:<28} {:<10} {:<10} {:<10} {:<12} {:<10} {:<10} {:<10}'.format(
            'Shot', 'Strategy', 'P@10', 'P@20', 'P@50', 'Recall@50',
            'Referral', 'Coverage', 'Sel_Risk')
        lines.append(H)
        lines.append('-' * 112)

        for shot in shots:
            val = df[(df['setting'] == 'common') & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]
            vp, vl = val['prob'].values, val['y_true'].values
            vu_raw = get_unc_raw(vp, val['uncertainty'].values, shot)
            vu = rank_scores(vu_raw)
            tau_p, tau_u = find_thresholds(vp, vu, vl)
            sorted_vu = np.sort(vu_raw)
            lines.append('  [thresholds] shot={}: tau_p={:.3f}, tau_u={:.3f} '
                         '(selected on validation)'.format(shot, tau_p, tau_u))

            test = df[(df['setting'] == setting) & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]

            all_metrics = []
            for seed in SEEDS:
                g = test[test[SEED_COL] == seed]
                tp_arr = g['prob'].values
                tl = g['y_true'].values
                n = len(tl)
                total_pos = tl.sum()
                tu_raw = get_unc_raw(tp_arr, g['uncertainty'].values, shot)
                tu = ecdf_map(tu_raw, sorted_vu)

                rng = np.random.RandomState(42 + seed)
                rand_ord = rng.permutation(n)
                prob_ord = np.argsort(-tp_arr)
                score = tp_arr * (1 - tu)
                ua_ord = np.argsort(-score)

                # ---- 四象限（统一语义）----
                cat = np.full(n, -1, dtype=int)
                for i in range(n):
                    if tp_arr[i] >= tau_p and tu[i] <= tau_u:
                        cat[i] = 0  # High-Priority   (automatic)
                    elif tp_arr[i] >= tau_p and tu[i] > tau_u:
                        cat[i] = 1  # Expert Review   (referred)
                    elif tp_arr[i] < tau_p and tu[i] > tau_u:
                        cat[i] = 2  # Deferred Review (referred)
                    else:
                        cat[i] = 3  # Low Priority    (automatic)

                referred_mask = (cat == 1) | (cat == 2)  # u > tau_u
                auto_mask = (cat == 0) | (cat == 3)      # u <= tau_u
                ua_ref = referred_mask.mean()
                ua_cov = auto_mask.mean()
                ua_sel = np.mean((tp_arr[auto_mask] >= tau_p).astype(int) != tl[auto_mask]) \
                    if auto_mask.sum() > 0 else 0

                # Probability-only (no referral; automatic = all)
                prob_ref, prob_cov = 0.0, 1.0
                prob_sel = np.mean((tp_arr >= tau_p).astype(int) != tl)

                # Random ranking: same mask as UA; selective risk = random-guessing error on auto set
                if auto_mask.sum() > 0:
                    pos_rate = tl[auto_mask].mean()
                    rand_risk = 1.0 - max(pos_rate, 1.0 - pos_rate)
                else:
                    rand_risk = 0.0

                def pk(ord_idx, K):
                    return tl[ord_idx[:K]].mean() if K <= n else tl[ord_idx].mean()

                def rk(ord_idx, K):
                    return tl[ord_idx[:K]].sum() / total_pos if total_pos > 0 and n >= K \
                        else (tl[ord_idx].sum() / total_pos if total_pos > 0 else 0)

                all_metrics.append({
                    'rand': {K: pk(rand_ord, K) for K in [10, 20, 50]},
                    'rand_R50': rk(rand_ord, 50),
                    'rand_ref': ua_ref, 'rand_cov': ua_cov, 'rand_risk': rand_risk,
                    'prob': {K: pk(prob_ord, K) for K in [10, 20, 50]},
                    'prob_R50': rk(prob_ord, 50),
                    'prob_ref': prob_ref, 'prob_cov': prob_cov, 'prob_sel': prob_sel,
                    'ua': {K: pk(ua_ord, K) for K in [10, 20, 50]},
                    'ua_R50': rk(ua_ord, 50),
                    'ua_ref': ua_ref, 'ua_cov': ua_cov, 'ua_sel': ua_sel,
                    'n_auto': int(auto_mask.sum()), 'n_referred': int(referred_mask.sum()),
                    'n': n,
                    # P1-9 (10.4)：四类 action 数量
                    'n_hp': int((cat == 0).sum()), 'n_er': int((cat == 1).sum()),
                    'n_dr': int((cat == 2).sum()), 'n_lp': int((cat == 3).sum()),
                })

            def avg(key):
                vals = [m[key] for m in all_metrics]
                if isinstance(vals[0], dict):
                    return {k: np.mean([v[k] for v in vals]) for k in vals[0]}
                return np.mean(vals)

            rows = [
                ('Random ranking', avg('rand'), avg('rand_R50'),
                 avg('rand_ref'), avg('rand_cov'), avg('rand_risk')),
                ('Probability only', avg('prob'), avg('prob_R50'),
                 avg('prob_ref'), avg('prob_cov'), avg('prob_sel')),
                ('Probability + uncertainty', avg('ua'), avg('ua_R50'),
                 avg('ua_ref'), avg('ua_cov'), avg('ua_sel')),
            ]
            for label, P_dict, R50, ref, cov, sel in rows:
                lines.append('{:<6} {:<28} {:<10} {:<10} {:<10} {:<12} {:<10} {:<10} {:<10}'.format(
                    shot, label, fmt(P_dict[10]), fmt(P_dict[20]), fmt(P_dict[50]),
                    fmt(R50), fmt(ref), fmt(cov), fmt(sel)))
            n_auto = avg('n_auto')
            n_ref = avg('n_referred')
            lines.append('  [counts] shot={}: mean automatic={:.0f}, referred={:.0f} '
                         'of {} candidates per seed'.format(shot, n_auto, n_ref, avg('n')))
            lines.append('  [action counts] High-Priority={:.1f} Expert-Referral={:.1f} '
                         'Deferred-Review={:.1f} Low-Priority={:.1f} (mean per seed; '
                         'fractions {:.1%}/{:.1%}/{:.1%}/{:.1%})'.format(
                             avg('n_hp'), avg('n_er'), avg('n_dr'), avg('n_lp'),
                             avg('n_hp') / avg('n'), avg('n_er') / avg('n'),
                             avg('n_dr') / avg('n'), avg('n_lp') / avg('n')))

        lines.append('-' * 112)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - Unified semantics: automatic = {high-priority, low-priority} (u <= tau_u);')
    lines.append('    referred = {expert referral, deferred review} (u > tau_u).')
    lines.append('  - u values are percentile ranks in [0,1]; test scores use the validation CDF.')
    lines.append('  - Random rows use the same referral mask as the UA strategy;')
    lines.append('    its selective risk is the random-guessing error on the automatic set.')
    lines.append('  - 1-shot signal u_entropy = H(p): confidence-derived baseline, NOT epistemic.')
    lines.append('  - 5-shot signal u_latent = SRAE latent dispersion score')
    lines.append('    (reconstruction-derived proxy; no KL-based posterior interpretation).')
    lines.append('  - Per-seed metrics averaged over 5 training seeds.')

    # ============================================================
    # Threshold sensitivity / risk-coverage curve + AURC
    # ============================================================
    lines.append('')
    lines.append('=' * 120)
    lines.append('RISK-COVERAGE CURVES AND AURC (threshold sensitivity over tau_u)')
    lines.append('=' * 120)
    for setting in settings:
        s_label = 'Fewer-shot (test)' if setting == 'fewer' else 'Rare-event (test2)'
        lines.append('')
        lines.append('--- {} ---'.format(s_label))
        for shot in shots:
            val = df[(df['setting'] == 'common') & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]
            vp, vl = val['prob'].values, val['y_true'].values
            vu_raw = get_unc_raw(vp, val['uncertainty'].values, shot)
            vu = rank_scores(vu_raw)
            tau_p, _ = find_thresholds(vp, vu, vl)
            sorted_vu = np.sort(vu_raw)
            test = df[(df['setting'] == setting) & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]
            curve = []
            for seed in SEEDS:
                g = test[test[SEED_COL] == seed]
                tp_arr = g['prob'].values
                tl = g['y_true'].values
                tu = ecdf_map(get_unc_raw(tp_arr, g['uncertainty'].values, shot), sorted_vu)
                for cand_tau_u in np.linspace(0.05, 0.95, 40):
                    auto = tu <= cand_tau_u
                    cov = auto.mean()
                    risk = np.mean((tp_arr[auto] >= tau_p).astype(int) != tl[auto]) \
                        if auto.sum() > 0 else 0
                    curve.append((cov, risk))
            curve = pd.DataFrame(curve, columns=['coverage', 'risk']).groupby('coverage').mean().reset_index()
            curve = curve.sort_values('coverage')
            aurc = np.trapezoid(curve['risk'].values, curve['coverage'].values)
            lines.append('  shot={}: AURC={:.4f}'.format(shot, aurc))
            for _, r in curve.iloc[np.linspace(0, len(curve) - 1, 6).astype(int)].iterrows():
                lines.append('    coverage={:.3f}  selective_risk={:.4f}'.format(r['coverage'], r['risk']))

    # ============================================================
    # Matched-coverage comparison (same coverage for both strategies)
    # ============================================================
    lines.append('')
    lines.append('=' * 120)
    lines.append('MATCHED-COVERAGE COMPARISON (same automatic coverage for both strategies)')
    lines.append('=' * 120)
    for setting in settings:
        s_label = 'Fewer-shot (test)' if setting == 'fewer' else 'Rare-event (test2)'
        lines.append('')
        lines.append('--- {} ---'.format(s_label))
        for shot in shots:
            val = df[(df['setting'] == 'common') & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]
            vp, vl = val['prob'].values, val['y_true'].values
            vu_raw = get_unc_raw(vp, val['uncertainty'].values, shot)
            vu = rank_scores(vu_raw)
            tau_p, _ = find_thresholds(vp, vu, vl)
            sorted_vu = np.sort(vu_raw)
            test = df[(df['setting'] == setting) & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]
            for cov_target in [0.30, 0.50, 0.70]:
                ua_risks, pb_risks = [], []
                for seed in SEEDS:
                    g = test[test[SEED_COL] == seed]
                    tp_arr = g['prob'].values
                    tl = g['y_true'].values
                    tu = ecdf_map(get_unc_raw(tp_arr, g['uncertainty'].values, shot), sorted_vu)
                    # UA: smallest tau_u achieving the target coverage
                    cand = np.linspace(tu.min(), tu.max(), 200)
                    best_u = tu.max()
                    for ct in cand:
                        if (tu <= ct).mean() >= cov_target:
                            best_u = ct
                            break
                    auto_ua = tu <= best_u
                    ua_risks.append(np.mean((tp_arr[auto_ua] >= tau_p).astype(int) != tl[auto_ua])
                                    if auto_ua.sum() > 0 else 0)
                    # Prob-only: keep top-K by prob (same automatic size)
                    k = max(1, int(cov_target * len(tl)))
                    top = np.argsort(-tp_arr)[:k]
                    pb_risks.append(np.mean((tp_arr[top] >= tau_p).astype(int) != tl[top]))
                pr, ur = np.mean(pb_risks), np.mean(ua_risks)
                red = (pr - ur) / pr * 100 if pr > 0 else 0
                lines.append('  shot={} coverage={:.2f}: prob-only risk={:.4f}, '
                             'UA risk={:.4f}, reduction={:+.1f}%'.format(
                                 shot, cov_target, pr, ur, red))

    # ============================================================
    # Fixed referral budget comparison
    # ============================================================
    lines.append('')
    lines.append('=' * 120)
    lines.append('FIXED REFERRAL BUDGET COMPARISON (same automatic size for both strategies)')
    lines.append('=' * 120)
    for setting in settings:
        s_label = 'Fewer-shot (test)' if setting == 'fewer' else 'Rare-event (test2)'
        lines.append('')
        lines.append('--- {} ---'.format(s_label))
        for shot in shots:
            val = df[(df['setting'] == 'common') & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]
            vp, vl = val['prob'].values, val['y_true'].values
            vu_raw = get_unc_raw(vp, val['uncertainty'].values, shot)
            vu = rank_scores(vu_raw)
            tau_p, _ = find_thresholds(vp, vu, vl)
            sorted_vu = np.sort(vu_raw)
            test = df[(df['setting'] == setting) & (df['shot'] == shot) & (df[SEED_COL].isin(SEEDS))]
            for budget in [0.10, 0.30, 0.50]:
                ua_risks, pb_risks = [], []
                for seed in SEEDS:
                    g = test[test[SEED_COL] == seed]
                    tp_arr = g['prob'].values
                    tl = g['y_true'].values
                    tu = ecdf_map(get_unc_raw(tp_arr, g['uncertainty'].values, shot), sorted_vu)
                    # UA: choose tau_u so that referred fraction <= budget
                    cand = np.linspace(tu.min(), tu.max(), 200)
                    for ct in cand:
                        if (tu > ct).mean() <= budget:
                            best_u = ct
                            break
                    else:
                        best_u = tu.max()
                    auto_ua = tu <= best_u
                    ua_risks.append(np.mean((tp_arr[auto_ua] >= tau_p).astype(int) != tl[auto_ua])
                                    if auto_ua.sum() > 0 else 0)
                    # Prob-only: keep the same number of automatic candidates (top by prob)
                    k = max(1, int(auto_ua.sum()))
                    top = np.argsort(-tp_arr)[:k]
                    pb_risks.append(np.mean((tp_arr[top] >= tau_p).astype(int) != tl[top]))
                lines.append('  shot={} budget={:.0%}: UA risk={:.4f}, '
                             'prob-only risk={:.4f}'.format(shot, budget,
                                                            np.mean(ua_risks), np.mean(pb_risks)))

    out = '\n'.join(lines) + '\n'
    print(out)
    os.makedirs('results', exist_ok=True)
    with open('results/table4_paper.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\nSaved to: results/table4_paper.txt')


if __name__ == '__main__':
    main()
