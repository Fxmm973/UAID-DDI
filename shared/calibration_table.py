#!/usr/bin/env python
# coding=utf-8
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
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
    return {'auroc': auroc, 'auprc': auprc, 'acc': float((g['y_pred'].values == labels).mean()),
            'f1_macro': event_macro_f1(g), 'ece': ece, 'brier': brier, 'nll': nll,
            'hce': h, 'hce_coverage': hcov, 'hce_count': hcnt,
            'n': int(len(g)), 'bin_counts': np.asarray(counts)}


def aggregate(seed_metrics):
    out = {}
    for k in seed_metrics[0]:
        vals = [m[k] for m in seed_metrics]
        if k == 'bin_counts':
            out[k] = np.sum(np.vstack(vals), axis=0)
        elif k == 'n':
            out[k] = int(np.sum(vals))
        elif k == 'hce_count':
            out[k] = int(np.sum(vals))
        else:
            out[k + '_mean'] = float(np.nanmean(vals))
            out[k + '_sd'] = float(np.nanstd(vals, ddof=1))
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
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ['prob', 'y_true', 'y_pred']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['y_true'] = df['y_true'].astype(int)
    df['y_pred'] = df['y_pred'].astype(int)
    if 'train_seed' not in df.columns and 'seed' in df.columns:
        df = df.rename(columns={'seed': 'train_seed'})

    dev = df[df['setting'] == args.dev_setting]
    test_settings = [s for s in ['fewer', 'rare'] if s in set(df['setting'])]

    rows = []
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

            reliability_diagram(g['prob'].values, g['y_true'].values,
                                axes[si][mi], title=f'{setting} - {m}')
        n = int(sub[sub['method'] == args.methods[0]]['n'].sum()) if 'n' in sub.columns else int(len(sub[sub['method'] == args.methods[0]]))
        rows.append({'setting': setting, 'method': '-', 'row': 'No-skill p=0.5',
                     'auroc_mean': 0.5, 'auroc_sd': 0.0, 'auprc_mean': 0.5, 'auprc_sd': 0.0,
                     'f1_macro_mean': 2 / 3, 'f1_macro_sd': 0.0, 'acc_mean': 0.5, 'acc_sd': 0.0,
                     'ece_mean': 0.0, 'ece_sd': 0.0, 'brier_mean': 0.25, 'brier_sd': 0.0,
                     'nll_mean': LN2, 'nll_sd': 0.0, 'hce_mean': np.nan, 'hce_sd': np.nan,
                     'hce_coverage_mean': 0.0, 'hce_coverage_sd': 0.0, 'hce_count': 0,
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
    cols = ['setting', 'row', 'auroc_mean', 'auroc_sd', 'auprc_mean', 'auprc_sd',
            'acc_mean', 'acc_sd', 'f1_macro_mean', 'f1_macro_sd',
            'brier_mean', 'brier_sd', 'nll_mean', 'nll_sd',
            'ece_mean', 'ece_sd', 'hce_mean', 'hce_sd',
            'hce_coverage_mean', 'hce_count', 'n']
    pd.set_option('display.width', 260)
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
