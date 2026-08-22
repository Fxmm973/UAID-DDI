#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import os
from sklearn import metrics


def compute_ece(confidences, predictions, labels, n_bins=10):
    confidences = np.clip(np.asarray(confidences, dtype=float), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(confidences, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(confidences)) * np.abs(labels[m].mean() - confidences[m].mean())
    return ece


def compute_brier(probs, labels):
    return np.mean((probs - labels) ** 2)


def compute_nll(probs, labels):
    probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    return -np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped))


def compute_high_conf_error(confidences, predictions, labels, threshold=0.9):
    high_conf = confidences > threshold
    if np.sum(high_conf) == 0:
        return 0.0
    return np.mean(predictions[high_conf] != labels[high_conf])


def compute_calibration_metrics(y_true, y_prob, y_pred):
    confidence = np.maximum(y_prob, 1 - y_prob)

    results = {}
    results['ECE'] = compute_ece(y_prob, y_pred, y_true, n_bins=10)
    results['Brier'] = compute_brier(y_prob, y_true)
    results['NLL'] = compute_nll(y_prob, y_true)
    results['Avg_Conf'] = np.mean(confidence)
    results['Acc'] = metrics.accuracy_score(y_true, y_pred)
    results['High_Conf_Error'] = compute_high_conf_error(confidence, y_pred, y_true, threshold=0.9)
    return results


def fmt(mean_val, std_val):
    return f'{mean_val:.4f}±{std_val:.4f}'


def process_dataframe(df, method_col='method'):
    if 'train_seed' in df.columns and 'seed' not in df.columns:
        df = df.rename(columns={'train_seed': 'seed'})
    all_metrics = []
    for (seed, setting, shot, method), group in df.groupby(['seed', 'setting', 'shot', method_col]):
        y_true = group['y_true'].values
        y_prob = group['prob'].values
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_calibration_metrics(y_true, y_prob, y_pred)
        m['seed'] = seed
        m['setting'] = setting
        m['shot'] = shot
        m['method'] = method
        m['N'] = len(group)
        all_metrics.append(m)
    return pd.DataFrame(all_metrics)


def aggregate_metrics(metrics_df):
    agg_cols = ['ECE', 'Brier', 'NLL', 'Avg_Conf', 'Acc', 'High_Conf_Error']
    summary = metrics_df.groupby(['setting', 'shot', 'method']).agg(
        {c: ['mean', 'std'] for c in agg_cols}
    ).reset_index()
    summary.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c
                       for c in summary.columns]
    return summary


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))

    dfs = {}

    phar_path = os.path.join(BASE, 'results/predictions/predictions_dataset1_PharDDIE.csv')
    if os.path.exists(phar_path):
        dfs['PharDDIE'] = pd.read_csv(phar_path)
        print(f'Loaded PharDDIE: {len(dfs["PharDDIE"])} samples')
        ph = dfs['PharDDIE']
        if 'train_seed' in ph.columns and 'seed' not in ph.columns:
            ph['seed'] = ph['train_seed']
        n_seeds = ph['seed'].nunique()
        print(f'[SEED-CHAIN] PharDDIE calibration CSV: {n_seeds} distinct seeds.')
        if n_seeds != 5:
            raise RuntimeError(
                f'Expected 5 training seeds in PharDDIE prediction CSV, found {n_seeds}. '
                f'Re-export with pharddie_export_full.py '
                f'(5 independent checkpoints required; no fallback allowed).')

    wo_path = os.path.join(BASE, 'results/predictions/predictions_dataset1_wo_uncertainty.csv')
    if os.path.exists(wo_path):
        dfs['PharDDIE w/o VAE'] = pd.read_csv(wo_path)
        print(f'Loaded PharDDIE w/o VAE: {len(dfs["PharDDIE w/o VAE"])} samples')

    zs_paths = [
        os.path.join(BASE, '..', 'EviDDIE', 'results', 'predictions', 'predictions_eviddie_new_ablation.csv'),
        os.path.join(BASE, 'results', 'predictions', 'predictions_eviddie_new_ablation.csv'),
    ]
    for p in zs_paths:
        if os.path.exists(p):
            dfs['Zero-shot'] = pd.read_csv(p)
            print(f'Loaded Zero-shot variants: {len(dfs["Zero-shot"])} samples from {p}')
            break

    all_results = []
    for label, df in dfs.items():
        metrics_df = process_dataframe(df)
        all_results.append(metrics_df)

    combined = pd.concat(all_results, ignore_index=True)
    summary = aggregate_metrics(combined)

    detail_csv = os.path.join(BASE, 'results/table3_complete_detail.csv')
    combined.to_csv(detail_csv, index=False, float_format='%.6f')
    print(f'\nDetail per seed saved to: {detail_csv}')

    setting_order = ['common', 'fewer', 'rare']
    shot_order = [0, 1, 5, 10]
    method_order_zs = ['Softmax baseline', 'EviDDIE w/o EVI', 'EviDDIE']
    method_order_fs = ['PharDDIE', 'PharDDIE w/o VAE']

    lines = []
    lines.append('=' * 120)
    lines.append('Table 3: Calibration Performance of the Predictive Module (COMPLETE)')
    lines.append('=' * 120)
    H = f'{"Setting":<10} {"Shot":<6} {"Method":<20} {"ECE":<18} {"Brier":<18} {"NLL":<18} {"Avg_Conf":<18} {"Acc":<18} {"HC_Err":<18}'
    lines.append(H)
    lines.append('-' * 120)

    agg_cols = ['ECE', 'Brier', 'NLL', 'Avg_Conf', 'Acc', 'High_Conf_Error']

    for setting in setting_order:
        for shot in shot_order:
            if shot == 0:
                methods = method_order_zs
            else:
                methods = method_order_fs

            for method in methods:
                row = summary[(summary['setting'] == setting) &
                              (summary['shot'] == shot) &
                              (summary['method'] == method)]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                vals = []
                for col in agg_cols:
                    m = row.get(f'{col}_mean', None)
                    s = row.get(f'{col}_std', None)
                    if m is not None and s is not None:
                        vals.append(f'{m:.4f}±{s:.4f}')
                    else:
                        vals.append('N/A')
                line = f'{setting:<10} {shot:<6} {method:<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18} {vals[3]:<18} {vals[4]:<18} {vals[5]:<18}'
                lines.append(line)
        lines.append('-' * 120)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - ECE = Expected Calibration Error (10 equal-width bins, paper Eq. ece).')
    lines.append('  - Brier = Brier Score (MSE).')
    lines.append('  - NLL = Negative Log-Likelihood.')
    lines.append('  - Avg_Conf = Average confidence = mean(max(p, 1-p)).')
    lines.append('  - Acc = Accuracy.')
    lines.append('  - HC_Err = High-Confidence Error Rate (conf > 0.9).')
    lines.append('  - Lower is better for ECE, Brier, NLL, HC_Err.')
    lines.append('  - "PharDDIE w/o VAE" = deterministic fc_direct head (no uncertainty).')
    lines.append('  - EviDDIE uses Dirichlet-based evidential uncertainty (EDL).')
    lines.append('=' * 120)

    out = '\n'.join(lines)
    print('\n' + out)

    txt_path = os.path.join(BASE, 'results/table3_complete.txt')
    csv_path = os.path.join(BASE, 'results/table3_complete.csv')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f'\nSaved to: {txt_path}')

    summary.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'Saved to: {csv_path}')


if __name__ == '__main__':
    main()
