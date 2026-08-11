#!/usr/bin/env python
# coding=utf-8
"""
Table 4: Agent-Assisted Prioritization — per paper description.
BOTH strategies use the VAE model (p and u from same model).
- "Probability only":  rank by p
- "Probability + uncertainty": rank by r = p(1-u)
Thresholds tau_p, tau_u selected on validation set (common).
Metrics reported on held-out test sets (fewer, rare).
"""
import pandas as pd, numpy as np, os

df = pd.read_csv('results/predictions/predictions_dataset1_PharDDIE.csv')
SEED_COL = 'train_seed' if 'train_seed' in df.columns else 'seed'
seeds_3 = [19940419, 20230801, 20240115, 20240520, 20240910]

def find_thresholds(probs, uncs, labels):
    """tau_p: max F1; tau_u: min selective risk with coverage >= 30%"""
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

    best_risk, best_tau_u = float('inf'), np.median(uncs)
    for tau in np.arange(np.percentile(uncs, 10), np.percentile(uncs, 90), 0.02):
        auto = uncs <= tau
        if auto.mean() < 0.30:
            continue
        risk = np.mean((probs[auto] >= best_tau_p).astype(int) != labels[auto]) if auto.sum() > 0 else 0
        if risk < best_risk:
            best_risk, best_tau_u = risk, tau
    return best_tau_p, best_tau_u

def compute_entropy(probs):
    """Confidence-derived uncertainty baseline (NOT epistemic uncertainty).
    H(p) = -p*log2(p) - (1-p)*log2(1-p). Max at p=0.5, min at p=0 or 1.
    This captures model confidence, not evidential uncertainty."""
    p = np.clip(probs, 1e-15, 1 - 1e-15)
    return (-p * np.log(p) - (1 - p) * np.log(1 - p)) / np.log(2)

def fmt(x):
    return '{:.4f}'.format(x)

def main():
    settings = ['fewer', 'rare']
    shots = [1, 5, 10]

    lines = []
    lines.append('=' * 120)
    lines.append('Table 4: Agent-Assisted Prioritization on Held-Out DDI Candidates')
    lines.append('Comparison: Probability-only ranking vs Uncertainty-aware ranking r = p(1-u)')
    lines.append('Both use VAE evidential model predictions; thresholds from common validation set.')
    lines.append('=' * 120)

    for setting in settings:
        s_label = 'Fewer-shot (test)' if setting == 'fewer' else 'Rare-event (test2)'
        lines.append('')
        lines.append('--- Setting: {} ---'.format(s_label))
        H = '{:<6} {:<30} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format(
            'Shot', 'Strategy', 'P@10', 'P@20', 'P@50', 'Recall@50',
            'Referral', 'Coverage', 'Sel_Risk')
        lines.append(H)
        lines.append('-' * 115)

        for shot in shots:
            # === Validation: thresholds from common set ===
            val = df[(df['setting'] == 'common') & (df['shot'] == shot) & (df[SEED_COL].isin(seeds_3))]
            vp, vl = val['prob'].values, val['y_true'].values
            if shot == 1:
                vu_raw = compute_entropy(vp)
            else:
                vu_raw = val['uncertainty'].values
            vu = (vu_raw - vu_raw.min()) / (vu_raw.max() - vu_raw.min()) if vu_raw.max() > vu_raw.min() else vu_raw
            tau_p, tau_u = find_thresholds(vp, vu, vl)

            # === Test set ===
            test = df[(df['setting'] == setting) & (df['shot'] == shot) & (df[SEED_COL].isin(seeds_3))]

            # Aggregate metrics over seeds
            all_metrics = []
            for seed in seeds_3:
                g = test[test[SEED_COL] == seed]
                tp_arr = g['prob'].values
                tl = g['y_true'].values
                n = len(tl)
                total_pos = tl.sum()

                # Uncertainty: VAE variance (shot>=5) or entropy (shot=1)
                if shot == 1:
                    tu_raw = compute_entropy(tp_arr)
                else:
                    tu_raw = g['uncertainty'].values
                tu_test = (tu_raw - tu_raw.min()) / (tu_raw.max() - tu_raw.min()) if tu_raw.max() > tu_raw.min() else tu_raw
                tu_test = np.clip(tu_test, 0, 1)

                # === Rankings ===
                # Random
                rng = np.random.RandomState(42 + seed)
                rand_ord = rng.permutation(n)

                # Probability-only: rank by p descending
                prob_ord = np.argsort(-tp_arr)

                # Uncertainty-aware: rank by r = p * (1-u) descending
                score = tp_arr * (1 - tu_test)
                ua_ord = np.argsort(-score)

                def pk(ord_idx, K):
                    return tl[ord_idx[:K]].mean() if K <= n else tl[ord_idx].mean()
                def rk(ord_idx, K):
                    return tl[ord_idx[:K]].sum() / total_pos if total_pos > 0 and n >= K else (tl[ord_idx].sum() / total_pos if total_pos > 0 else 0)

                # UA decision categories (4-quadrant)
                cat = np.full(n, -1, dtype=int)
                for i in range(n):
                    if tp_arr[i] >= tau_p and tu_test[i] <= tau_u:
                        cat[i] = 0  # High-Priority
                    elif tp_arr[i] >= tau_p and tu_test[i] > tau_u:
                        cat[i] = 1  # Expert Review
                    elif tp_arr[i] < tau_p and tu_test[i] > tau_u:
                        cat[i] = 2  # Defer
                    else:
                        cat[i] = 3  # Low Priority

                # Probability-only: no referral (no u)
                prob_ref = 0.0
                prob_cov = 1.0
                prob_sel = np.mean((tp_arr >= tau_p).astype(int) != tl)

                # UA: referral / coverage / sel_risk
                ua_ref = np.mean(cat == 1)
                auto = cat != 1
                ua_cov = auto.mean()
                ua_sel = np.mean((tp_arr[auto] >= tau_p).astype(int) != tl[auto]) if auto.sum() > 0 else 0

                all_metrics.append({
                    'rand': {K: pk(rand_ord, K) for K in [10, 20, 50]},
                    'rand_R50': rk(rand_ord, 50),
                    'prob': {K: pk(prob_ord, K) for K in [10, 20, 50]},
                    'prob_R50': rk(prob_ord, 50),
                    'prob_ref': prob_ref, 'prob_cov': prob_cov, 'prob_sel': prob_sel,
                    'ua': {K: pk(ua_ord, K) for K in [10, 20, 50]},
                    'ua_R50': rk(ua_ord, 50),
                    'ua_ref': ua_ref, 'ua_cov': ua_cov, 'ua_sel': ua_sel,
                })

            # Average over seeds
            def avg(key):
                vals = [m[key] for m in all_metrics]
                if isinstance(vals[0], dict):
                    return {k: np.mean([v[k] for v in vals]) for k in vals[0]}
                return np.mean(vals)

            rand_P = avg('rand'); rand_R50 = avg('rand_R50')
            prob_P = avg('prob'); prob_R50 = avg('prob_R50')
            prob_ref = avg('prob_ref'); prob_cov = avg('prob_cov'); prob_sel = avg('prob_sel')
            ua_P = avg('ua'); ua_R50 = avg('ua_R50')
            ua_ref = avg('ua_ref'); ua_cov = avg('ua_cov'); ua_sel = avg('ua_sel')

            for label, P_dict, R50, ref, cov, sel in [
                ('Random ranking', rand_P, rand_R50, None, None, None),
                ('Probability only', prob_P, prob_R50, prob_ref, prob_cov, prob_sel),
                ('Probability + uncertainty', ua_P, ua_R50, ua_ref, ua_cov, ua_sel),
            ]:
                if ref is None:
                    extras = ['--', '--', '--']
                else:
                    extras = [fmt(ref), fmt(cov), fmt(sel)]
                line = '{:<6} {:<30} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format(
                    shot, label, fmt(P_dict[10]), fmt(P_dict[20]), fmt(P_dict[50]),
                    fmt(R50), extras[0], extras[1], extras[2])
                lines.append(line)

            # Also show delta P@K and Sel_Risk reduction
            delta_p10 = prob_P[10] - ua_P[10]
            delta_sel = (prob_sel - ua_sel) / prob_sel * 100 if prob_sel > 0 else 0
            lines.append('  -> P@10 delta: {:.4f}  |  Sel_Risk reduction: {:.1f}%'.format(delta_p10, delta_sel))

        lines.append('-' * 115)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - Both strategies use the SAME VAE evidential model predictions.')
    lines.append('  - "Probability only" ranks by p; "Probability + uncertainty" ranks by r = p(1-u).')
    lines.append('  - tau_p and tau_u selected on common validation set via max-F1 / min-selective-risk.')
    lines.append('  - Referral = fraction sent to Expert Review (p>=tau_p AND u>tau_u).')
    lines.append('  - Coverage = fraction auto-decided = 1 - Referral (NOT 1 - Expert Review rate).')
    lines.append('  - Selective Risk = error rate among auto-decided predictions.')
    lines.append('  - For shot=1: entropy H(p) used as CONFIDENCE-DERIVED baseline (NOT epistemic).')
    lines.append('    VAE variance is unreliable with only K=1 support example.')
    lines.append('  - For shot>=5: normalized VAE latent variance used as uncertainty.')
    lines.append('  - Per-seed metrics averaged over 3 seeds (2024, 2025, 2026).')
    lines.append('=' * 120)

    # ============================================================
    # Matched-Coverage Risk-Coverage Analysis (reviewer request)
    # ============================================================
    lines.append('')
    lines.append('=' * 120)
    lines.append('MATCHED-COVERAGE RISK-COVERAGE ANALYSIS')
    lines.append('Compares selective risk at matched coverage levels for fair comparison.')
    lines.append('Recall: "deferred review" IS a referral action - it is NOT automatic coverage.')
    lines.append('=' * 120)

    for setting in settings:
        s_label = 'Fewer-shot (test)' if setting == 'fewer' else 'Rare-event (test2)'
        lines.append('')
        lines.append(f'--- {s_label} ---')
        H2 = f'{"Shot":<6} {"Coverage":<10} {"Prob-only Risk":<16} {"UA Risk":<16} {"Risk Reduction":<16}'
        lines.append(H2)
        lines.append('-' * 70)

        for shot in shots:
            val = df[(df['setting'] == 'common') & (df['shot'] == shot) & (df[SEED_COL].isin(seeds_3))]
            vp, vl = val['prob'].values, val['y_true'].values
            if shot == 1:
                vu_raw = compute_entropy(vp)
            else:
                vu_raw = val['uncertainty'].values
            vu = (vu_raw - vu_raw.min()) / (vu_raw.max() - vu_raw.min()) if vu_raw.max() > vu_raw.min() else vu_raw
            tau_p, tau_u = find_thresholds(vp, vu, vl)

            test = df[(df['setting'] == setting) & (df['shot'] == shot) & (df[SEED_COL].isin(seeds_3))]

            cov_levels = [0.30, 0.50, 0.70, 0.90, 1.00]
            matched = {c: {'prob_risk': [], 'ua_risk': []} for c in cov_levels}

            for seed in seeds_3:
                g = test[test[SEED_COL] == seed]
                tp_arr = g['prob'].values
                tl = g['y_true'].values
                if shot == 1:
                    tu_raw = compute_entropy(tp_arr)
                else:
                    tu_raw = g['uncertainty'].values
                tu_test = (tu_raw - tu_raw.min()) / (tu_raw.max() - tu_raw.min()) if tu_raw.max() > tu_raw.min() else tu_raw

                for cov_target in cov_levels:
                    best_u_tau = tu_test.max()
                    for candidate_tau_u in np.linspace(tu_test.min(), tu_test.max(), 200):
                        auto_mask = tu_test <= candidate_tau_u
                        if auto_mask.mean() >= cov_target:
                            best_u_tau = candidate_tau_u
                            break
                    auto_ua = tu_test <= best_u_tau
                    ua_risk = np.mean((tp_arr[auto_ua] >= tau_p).astype(int) != tl[auto_ua]) if auto_ua.sum() > 0 else 0
                    matched[cov_target]['ua_risk'].append(ua_risk)

                    n_keep = max(1, int(cov_target * len(tl)))
                    prob_ord = np.argsort(-tp_arr)
                    prob_auto = np.zeros(len(tl), dtype=bool)
                    prob_auto[prob_ord[:n_keep]] = True
                    prob_risk = np.mean((tp_arr[prob_auto] >= tau_p).astype(int) != tl[prob_auto])
                    matched[cov_target]['prob_risk'].append(prob_risk)

            for cov_target in cov_levels:
                pr_mean = np.mean(matched[cov_target]['prob_risk'])
                pr_std = np.std(matched[cov_target]['prob_risk'])
                ur_mean = np.mean(matched[cov_target]['ua_risk'])
                ur_std = np.std(matched[cov_target]['ua_risk'])
                reduction = (pr_mean - ur_mean) / pr_mean * 100 if pr_mean > 0 else 0
                lines.append(f'{shot:<6} {cov_target:<10.2f} '
                             f'{pr_mean:.4f}+/-{pr_std:.4f}   '
                             f'{ur_mean:.4f}+/-{ur_std:.4f}   '
                             f'{reduction:+.1f}%')

        lines.append('-' * 70)

    lines.append('')
    lines.append('Matched-Coverage Notes:')
    lines.append('  - "Coverage" = fraction of candidates auto-decided (NOT referred).')
    lines.append('  - At matched coverage, we compare selective risk FAIRLY.')
    lines.append('  - For shot=1: "uncertainty" = entropy H(p) - a CONFIDENCE-DERIVED')
    lines.append('    baseline, NOT epistemic uncertainty. VAE variance is unreliable at K=1.')
    lines.append('  - For shot>=5: uncertainty = normalized VAE latent variance.')
    lines.append('  - Deferred review (p<tau_p, u>=tau_u) IS a referral action.')
    lines.append('=' * 120)

    out = '\n'.join(lines)
    print(out)

    os.makedirs('results', exist_ok=True)
    with open('results/table4_paper.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\nSaved to: results/table4_paper.txt')


if __name__ == '__main__':
    main()
