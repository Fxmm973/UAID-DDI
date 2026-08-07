#!/usr/bin/env Python
# coding=utf-8
"""
Table 3: Agent-Assisted Prioritization.
Adds matched-coverage selective risk comparison (M8 fix).
"""
import pandas as pd, numpy as np, os

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


def compute_risk_coverage_curve(probs, uncs, labels, tau_p, n_points=20):
    """Compute selective risk at different coverage levels (varying tau_u)."""
    n = len(labels)
    tau_u_range = np.percentile(uncs, np.linspace(5, 95, n_points))
    points = []
    for tau_u in tau_u_range:
        auto = uncs <= tau_u
        coverage = auto.mean()
        if coverage < 0.05 or auto.sum() == 0:
            continue
        risk = np.mean((probs[auto] >= tau_p).astype(int) != labels[auto])
        points.append((coverage, risk))
    return sorted(points)


def matched_coverage_risk(probs, labels, target_coverage):
    """Prob-only selective risk at target coverage (keep top-K by prob)."""
    n = len(labels)
    k = max(1, int(n * target_coverage))
    top_idx = np.argsort(-probs)[:k]  # top-K by probability
    tau_p_match = np.median(probs[top_idx])  # use median as threshold
    risk = np.mean((probs[top_idx] >= tau_p_match).astype(int) != labels[top_idx]) if len(top_idx) > 0 else 0
    return risk, tau_p_match, k


def f3(x):
    return f'{x:.4f}'


def main():
    df_phar = pd.read_csv('results/predictions/predictions_dataset1_PharDDIE.csv')

    lines = []
    lines.append('=' * 110)
    lines.append('Table 3: Agent-Assisted Prioritization on Held-Out DDI Candidates.')
    lines.append('Uncertainty source: VAE latent variance (normalized per shot/setting).')
    lines.append('1-shot uses entropy fallback (VAE variance unreliable in low-data regime).')
    lines.append('=' * 110)

    for setting, s_label in [('fewer', 'fewer (test)'), ('rare', 'rare (test2)')]:
        lines.append(f'\n--- Setting: {s_label} ---')
        H = (f"{'Shot':<6} {'Strategy':<22} {'P@10':<10} {'P@20':<10} {'P@50':<10} "
             f"{'Recall@50':<12} {'Referral':<10} {'Coverage':<10} {'Sel_Risk':<10} {'HC_Err':<10}")
        lines.append(H)
        lines.append('-' * 100)

        for shot in [1, 5, 10]:
            # ---- validation: thresholds ----
            val = df_phar[(df_phar['setting'] == 'common') & (df_phar['shot'] == shot)]
            vp, vl = val['prob'].values, val['y_true'].values
            if shot == 1:
                vu_raw = -vp * np.log(np.clip(vp, 1e-15, 1)) - (1 - vp) * np.log(np.clip(1 - vp, 1e-15, 1))
                vu_raw = vu_raw / np.log(2)
            else:
                vu_raw = val['uncertainty'].values
            vu = (vu_raw - vu_raw.min()) / (vu_raw.max() - vu_raw.min()) if vu_raw.max() > vu_raw.min() else vu_raw
            tau_p, tau_u = find_thresholds(vp, vu, vl)

            # ---- test set ----
            test = df_phar[(df_phar['setting'] == setting) & (df_phar['shot'] == shot)]
            tp_arr, tl = test['prob'].values, test['y_true'].values
            if shot == 1:
                tu_raw = -tp_arr * np.log(np.clip(tp_arr, 1e-15, 1)) - (1 - tp_arr) * np.log(np.clip(1 - tp_arr, 1e-15, 1))
                tu_raw = tu_raw / np.log(2)
            else:
                tu_raw = test['uncertainty'].values
            tu = (tu_raw - vu_raw.min()) / (vu_raw.max() - vu_raw.min()) if vu_raw.max() > vu_raw.min() else tu_raw
            tu = np.clip(tu, 0, 1)

            n = len(tl)
            total_pos = tl.sum()

            # Rankings
            rng = np.random.RandomState(42)
            rand_ord = rng.permutation(n)
            prob_ord = np.argsort(-tp_arr)
            score = tp_arr * (1 - tu)
            ua_ord = np.argsort(-score)

            # UA 四象限
            cat = np.full(n, -1, dtype=int)
            for i in range(n):
                if tp_arr[i] >= tau_p and tu[i] <= tau_u:
                    cat[i] = 0  # High-Priority
                elif tp_arr[i] >= tau_p and tu[i] > tau_u:
                    cat[i] = 1  # Expert Review
                elif tp_arr[i] < tau_p and tu[i] > tau_u:
                    cat[i] = 2  # Defer
                else:
                    cat[i] = 3  # Low Priority

            # Prob-only 指标（不依赖 uncertainty）
            prob_hc = np.maximum(tp_arr, 1 - tp_arr) > 0.9
            prob_sel = np.mean((tp_arr >= tau_p).astype(int) != tl)
            prob_hc_err = np.mean((tp_arr[prob_hc] >= tau_p).astype(int) != tl[prob_hc]) if prob_hc.sum() > 0 else 0

            # UA 指标
            ua_ref = np.mean(cat == 1)
            auto = cat != 1
            ua_cov = auto.mean()
            ua_sel = np.mean((tp_arr[auto] >= tau_p).astype(int) != tl[auto]) if auto.sum() > 0 else 0
            hp = cat == 0
            ua_hc = np.mean((tp_arr[hp] >= tau_p).astype(int) != tl[hp]) if hp.sum() > 0 else 0

            # ---- M8: Matched-coverage comparison ----
            if ua_cov > 0 and ua_cov < 1.0:
                prob_matched_risk, prob_matched_tau, prob_matched_k = matched_coverage_risk(tp_arr, tl, ua_cov)
            else:
                prob_matched_risk = prob_sel

            # P@K / Recall@K
            def pk(ord_idx, K):
                return tl[ord_idx[:K]].mean() if K <= n else tl[ord_idx].mean()
            def rk(ord_idx, K):
                return tl[ord_idx[:K]].sum() / total_pos if total_pos > 0 and n >= K else (tl[ord_idx].sum() / total_pos if total_pos > 0 else 0)

            # ---- Build rows ----
            for s_name, is_rand, is_prob, is_ua in [
                ('Random', True, False, False),
                ('Probability-only', False, True, False),
                ('Uncertainty-aware', False, False, True),
            ]:
                if is_rand:
                    vals = [f3(pk(rand_ord, K)) for K in [10, 20, 50]]
                    vals += [f3(rk(rand_ord, 50)), '--', '--', '--', '--']
                elif is_prob:
                    vals = [f3(pk(prob_ord, K)) for K in [10, 20, 50]]
                    vals += [f3(rk(prob_ord, 50)), '0.0000', '1.0000',
                             f3(prob_sel), f3(prob_hc_err)]
                else:  # UA
                    vals = [f3(pk(ua_ord, K)) for K in [10, 20, 50]]
                    vals += [f3(rk(ua_ord, 50)), f3(ua_ref), f3(ua_cov),
                             f3(ua_sel), f3(ua_hc)]

                lines.append(f'{shot:<6} {s_name:<22} {vals[0]:<10} {vals[1]:<10} {vals[2]:<10} '
                             f'{vals[3]:<12} {vals[4]:<10} {vals[5]:<10} {vals[6]:<10} {vals[7]:<10}')

            # ---- M8: Print matched-coverage note ----
            if ua_cov > 0 and ua_cov < 1.0:
                lines.append(f'        Matched-coverage comparison at cov={f3(ua_cov)}:')
                lines.append(f'        Probability-only selective risk: {f3(prob_matched_risk)}')
                lines.append(f'        Uncertainty-aware selective risk: {f3(ua_sel)}')
                lines.append(f'        Risk reduction: {f3(prob_matched_risk - ua_sel)}')

        lines.append('-' * 100)

    lines.append('')
    lines.append('Decision categories (Uncertainty-aware):')
    lines.append('  High-Priority: p >= tau_p AND u <= tau_u')
    lines.append('  Expert Review: p >= tau_p AND u >  tau_u')
    lines.append('  Defer:         p <  tau_p AND u >  tau_u')
    lines.append('  Low Priority:  p <  tau_p AND u <= tau_u')
    lines.append('')
    lines.append('Matched-coverage: Prob-only selects top-K by prob (K = UA coverage * N),')
    lines.append('  uses median prob as threshold. Compares selective risk at same operating point.')
    lines.append('')

    out = '\n'.join(lines)
    print(out)
    os.makedirs('results', exist_ok=True)
    with open('results/table3_final.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\nSaved to results/table3_final.txt')


if __name__ == '__main__':
    main()
