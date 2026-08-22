#!/usr/bin/env python
# coding=utf-8
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import ttest_rel, t as tdist
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HIGH_CONF = 0.9
LN2 = float(np.log(2))


def ece_brier_nll(probs, labels, n_bins=10):
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(p, bins[1:-1])
    ece, counts = 0.0, []
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            counts.append(0)
            continue
        acc = labels[m].mean()
        conf = p[m].mean()
        ece += (m.sum() / len(p)) * abs(acc - conf)
        counts.append(int(m.sum()))
    brier = float(np.mean((p - labels) ** 2))
    nll = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    return ece, brier, nll, counts


def hce(probs, labels, tau=HIGH_CONF):
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)

    conf = np.maximum(probs, 1 - probs)
    mask = conf >= tau

    if mask.sum() == 0:
        return np.nan, 0.0, 0

    pred = (probs >= 0.5).astype(int)
    err = (pred[mask] != labels[mask]).mean()

    return float(err), float(mask.mean()), int(mask.sum())


def lin_cal(probs, labels, n_bins=10):
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = (p > bins[b]) & (p <= bins[b + 1])
        if m.sum() > 0:
            xs.append(p[m].mean())
            ys.append(labels[m].mean())
    if len(xs) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(intercept), float(slope)


def event_macro_f1(g):
    f1s = []
    for ev, sub in g.groupby('event_type'):
        tp = ((sub['y_pred'] == 1) & (sub['y_true'] == 1)).sum()
        fp = ((sub['y_pred'] == 1) & (sub['y_true'] == 0)).sum()
        fn = ((sub['y_pred'] == 0) & (sub['y_true'] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else np.nan


def per_seed_metrics(g):
    probs, labels = g['prob'].values, g['y_true'].values
    auroc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else np.nan
    auprc = average_precision_score(labels, probs)
    ece, brier, nll, counts = ece_brier_nll(probs, labels)
    h, hcov, hcnt = hce(probs, labels)
    intercept, slope = lin_cal(probs, labels)
    conf = np.maximum(probs, 1 - probs)
    hmask = conf >= HIGH_CONF
    hce_err = int(((probs[hmask] >= 0.5).astype(int) != labels[hmask]).sum()) if hmask.sum() else 0
    return {'auroc': auroc, 'auprc': auprc, 'acc': float((g['y_pred'].values == labels).mean()),
            'f1_macro': event_macro_f1(g), 'ece': ece, 'brier': brier, 'nll': nll,
            'hce': h, 'hce_coverage': hcov, 'hce_count': hcnt, 'hce_err': hce_err,
            'intercept': intercept, 'slope': slope,
            'n': int(len(g)), 'bin_counts': np.asarray(counts)}


def aggregate(seed_metrics):
    out = {}
    n_seeds = len(seed_metrics)
    for k in seed_metrics[0]:
        vals = [m[k] for m in seed_metrics]
        if k == 'bin_counts':
            out[k] = np.sum(np.vstack(vals), axis=0)
        elif k in ('n', 'hce_count', 'hce_err'):
            out[k] = int(np.sum(vals))
        else:
            out[k + '_mean'] = float(np.nanmean(vals))
            out[k + '_sd'] = float(np.nanstd(vals, ddof=1))
    tval = tdist.ppf(0.975, n_seeds - 1) if n_seeds > 1 else np.nan
    for m in ('auroc', 'brier', 'nll', 'ece'):
        se = out[m + '_sd'] / np.sqrt(n_seeds)
        out[m + '_ci95_low'] = float(out[m + '_mean'] - tval * se)
        out[m + '_ci95_high'] = float(out[m + '_mean'] + tval * se)
    # intercept/slope 的 SD 使用 ddof=0（与论文表格的呈现口径一致）
    for m in ('intercept', 'slope'):
        vals = [x[m] for x in seed_metrics]
        out[m + '_sd'] = float(np.nanstd(vals, ddof=0))
    out['hce_cov_pooled'] = out['hce_count'] / out['n'] if out.get('n') else np.nan
    return out


def fit_temperature(probs, labels):
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    l = np.log(p / (1 - p))

    def nll(T):
        z = l / T
        q = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
        return float(-np.mean(labels * np.log(q) + (1 - labels) * np.log(1 - q)))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return res.x


def apply_temperature(probs, T):
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    z = np.log(p / (1 - p)) / T
    return 1 / (1 + np.exp(-z))


def reliability_diagram(probs, labels, ax, title, n_bins=10):
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(p, bins[1:-1])
    centers, accs, counts = [], [], []
    for b in range(n_bins):
        m = ids == b
        if m.sum() < 1:
            continue
        centers.append((bins[b] + bins[b + 1]) / 2)
        accs.append(labels[m].mean())
        counts.append(int(m.sum()))
    centers, accs, counts = map(np.asarray, (centers, accs, counts))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax.plot(centers, accs, 'o-', lw=2, color='tab:blue', label='Observed')
    for x, y, c in zip(centers, accs, counts):
        ax.annotate(str(c), (x, y), textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=8, color='gray')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed positive fraction')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='results/calibration_table.csv')
    ap.add_argument('--fig', default='results/reliability_diagram.png')
    ap.add_argument('--methods', nargs='+',
                    default=['EviDDIE', 'Softmax baseline', 'EviDDIE w/o EVI'])
    ap.add_argument('--dev-setting', default='common')
    ap.add_argument('--settings', nargs='+', default=None,
                    help='held-out settings to evaluate (default: fewer rare)')
    ap.add_argument('--shot', type=int, default=None,
                    help='filter rows by shot (for PharDDIE per-shot rows)')
    ap.add_argument('--paired', default=None,
                    help='paired native-vs-TempScale CSV output (default: <out>_paired.csv)')
    ap.add_argument('--detail', default=None,
                    help='per-seed detail CSV output (default: <out>_detail.csv)')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ['prob', 'y_true', 'y_pred']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['y_true'] = df['y_true'].astype(int)
    df['y_pred'] = df['y_pred'].astype(int)
    if 'train_seed' not in df.columns and 'seed' in df.columns:
        df = df.rename(columns={'seed': 'train_seed'})
    if args.shot is not None and 'shot' in df.columns:
        df = df[df['shot'] == args.shot]

    dev = df[df['setting'] == args.dev_setting]
    wanted = args.settings if args.settings else ['fewer', 'rare']
    test_settings = [s for s in wanted if s in set(df['setting'])]

    paired_csv = args.paired or args.out.replace('.csv', '_paired.csv')
    detail_csv = args.detail or args.out.replace('.csv', '_detail.csv')

    rows = []
    detail_rows = []
    paired_rows = []
    seed_Ts = {m: {} for m in args.methods}
    for m in args.methods:
        for s, g in dev[dev['method'] == m].groupby('train_seed'):
            seed_Ts[m][s] = fit_temperature(g['prob'].values, g['y_true'].values)

    fig, axes = plt.subplots(len(test_settings), len(args.methods),
                             figsize=(4 * len(args.methods), 3.2 * len(test_settings)),
                             squeeze=False)
    for si, setting in enumerate(test_settings):
        sub = df[df['setting'] == setting]
        for mi, m in enumerate(args.methods):
            g = sub[sub['method'] == m]
            if g.empty:
                continue
            seed_metrics = [per_seed_metrics(sg) for _, sg in g.groupby('train_seed')]
            row = aggregate(seed_metrics)
            row.update({'setting': setting, 'method': m, 'row': m})
            rows.append(row)
            for seed_val, sm in zip(g['train_seed'].unique(), seed_metrics):
                d = dict(sm)
                d.update({'setting': setting, 'method': m, 'row': m,
                          'train_seed': seed_val, 'scaled': False})
                detail_rows.append(d)

            ts_metrics = []
            for s, sg in g.groupby('train_seed'):
                if s not in seed_Ts[m]:
                    continue
                sg = sg.copy()
                sg['prob'] = apply_temperature(sg['prob'].values, seed_Ts[m][s])
                sg['y_pred'] = (sg['prob'] >= 0.5).astype(int)
                ts_metrics.append(per_seed_metrics(sg))
            if ts_metrics:
                rowt = aggregate(ts_metrics)
                rowt.update({'setting': setting, 'method': m, 'row': m + ' + TempScale'})
                rows.append(rowt)
                for seed_val, sm in zip(g['train_seed'].unique(), ts_metrics):
                    d = dict(sm)
                    d.update({'setting': setting, 'method': m, 'row': m + ' + TempScale',
                              'train_seed': seed_val, 'scaled': True})
                    detail_rows.append(d)
                for metric in ('ece', 'brier', 'nll'):
                    a = [sm[metric] for sm in seed_metrics]
                    b = [sm[metric] for sm in ts_metrics]
                    tval, pval = ttest_rel(a, b)
                    paired_rows.append({'setting': setting, 'method': m, 'metric': metric,
                                        'native_mean': float(np.mean(a)),
                                        'ts_mean': float(np.mean(b)),
                                        'diff': float(np.mean(a) - np.mean(b)),
                                        'p_paired_t': float(pval)})

            reliability_diagram(g['prob'].values, g['y_true'].values,
                                axes[si][mi], title=f'{setting} - {m}')
        n = int(sub[sub['method'] == args.methods[0]]['n'].sum()) if 'n' in sub.columns else int(len(sub[sub['method'] == args.methods[0]]))
        rows.append({'setting': setting, 'method': '-', 'row': 'No-skill p=0.5',
                     'auroc_mean': 0.5, 'auroc_sd': 0.0, 'auprc_mean': 0.5, 'auprc_sd': 0.0,
                     'f1_macro_mean': 2 / 3, 'f1_macro_sd': 0.0, 'acc_mean': 0.5, 'acc_sd': 0.0,
                     'ece_mean': 0.0, 'ece_sd': 0.0, 'brier_mean': 0.25, 'brier_sd': 0.0,
                     'nll_mean': LN2, 'nll_sd': 0.0, 'hce_mean': np.nan, 'hce_sd': np.nan,
                     'hce_coverage_mean': 0.0, 'hce_coverage_sd': 0.0, 'hce_count': 0,
                     'hce_err': 0, 'hce_cov_pooled': 0.0,
                     'intercept_mean': np.nan, 'intercept_sd': np.nan,
                     'slope_mean': np.nan, 'slope_sd': np.nan,
                     'auroc_ci95_low': np.nan, 'auroc_ci95_high': np.nan,
                     'brier_ci95_low': np.nan, 'brier_ci95_high': np.nan,
                     'nll_ci95_low': np.nan, 'nll_ci95_high': np.nan,
                     'ece_ci95_low': np.nan, 'ece_ci95_high': np.nan,
                     'n': n, 'bin_counts': None})

    for ax in axes.flat:
        if not ax.has_data():
            fig.delaxes(ax)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=200)
    print(f'[saved] {args.fig}')

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f'[saved] {args.out} ({len(out)} rows)')

    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)
    print(f'[saved] {detail_csv} ({len(detail_rows)} per-seed rows)')

    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(paired_csv, index=False)
        print(f'[saved] {paired_csv} ({len(paired_rows)} paired rows)')
        print('\n[P0-5-PAIRED] native vs TempScale (paired t over seeds)')
        for r in paired_rows:
            print(f'  {r["setting"]:6s} {r["method"]:22s} {r["metric"]:6s} '
                  f'native {r["native_mean"]:.4f} -> TS {r["ts_mean"]:.4f} '
                  f'diff {r["diff"]:+.4f} p={r["p_paired_t"]:.3f}')
    cols = ['setting', 'row', 'auroc_mean', 'auroc_sd', 'auroc_ci95_low', 'auroc_ci95_high',
            'auprc_mean', 'auprc_sd', 'f1_macro_mean', 'f1_macro_sd',
            'brier_mean', 'brier_sd', 'brier_ci95_low', 'brier_ci95_high',
            'nll_mean', 'nll_sd', 'nll_ci95_low', 'nll_ci95_high',
            'intercept_mean', 'intercept_sd', 'slope_mean', 'slope_sd',
            'ece_mean', 'ece_sd', 'ece_ci95_low', 'ece_ci95_high',
            'hce_mean', 'hce_sd', 'hce_err', 'hce_count', 'hce_cov_pooled', 'n']
    pd.set_option('display.width', 300)
    print(out[cols].round(4).to_string(index=False))

    print('\n[P0-2-STRICT-VERDICT] per (setting, method) against no-skill p=0.5')
    for _, r in out.iterrows():
        if r.get('row') not in args.methods:
            continue
        pass_proper = (r['brier_mean'] < 0.25) and (r['nll_mean'] < LN2)
        pass_disc = r['auroc_mean'] > 0.55
        verdict = 'PASS' if (pass_proper and pass_disc) else 'FAIL'
        print(f'  {r["setting"]:6s} {r["row"]:22s} brier={r["brier_mean"]:.4f}+-{r["brier_sd"]:.4f} '
              f'nll={r["nll_mean"]:.4f}+-{r["nll_sd"]:.4f} auroc={r["auroc_mean"]:.4f}+-{r["auroc_sd"]:.4f} '
              f'| proper-score: {"PASS" if pass_proper else "FAIL"} | discrimination: '
              f'{"PASS" if pass_disc else "FAIL"} | verdict: {verdict}')

    print('\n[P0-2-BALANCE] 1:1 positive:negative pairing check per (setting, method)')
    for s in test_settings:
        sub = df[df['setting'] == s]
        for m in args.methods:
            g = sub[sub['method'] == m]
            if g.empty:
                continue
            pos = int(g['y_true'].sum())
            neg = len(g) - pos
            ratio = pos / max(neg, 1)
            flag = 'OK' if abs(ratio - 1.0) < 0.05 else 'WARNING'
            print(f'  {s:6s} {m:22s} pos={pos} neg={neg} ratio={ratio:.4f} [{flag}]')

    for m in args.methods:
        Ts = list(seed_Ts[m].values())
        if Ts:
            print(f'[temperature] {m}: per-seed T = '
                  f'{", ".join(f"{v:.4f}" for v in Ts)} (mean {np.mean(Ts):.4f})')


if __name__ == '__main__':
    main()
