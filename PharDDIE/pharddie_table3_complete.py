#!/usr/bin/env python
# coding=utf-8
"""
Complete Table 3: Calibration Performance — ALL methods.
Reads all prediction CSVs and computes calibration metrics consistently.
"""
import pandas as pd
import numpy as np
import os
from sklearn import metrics


# ============================================================
# Calibration metrics (EXACT same as pharddie_table3.py)
# ============================================================

def compute_ece(confidences, predictions, labels, n_bins=15):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            acc_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += (bin_size / len(confidences)) * np.abs(acc_in_bin - avg_conf_in_bin)
    return ece


def compute_brier(probs, labels):
    """Brier Score (MSE between predicted prob and true label)"""
    return np.mean((probs - labels) ** 2)


def compute_nll(probs, labels):
    """Negative Log-Likelihood"""
    probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    return -np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped))


def compute_high_conf_error(confidences, predictions, labels, threshold=0.9):
    """High-confidence error rate: fraction of high-conf predictions that are wrong"""
    high_conf = confidences > threshold
    if np.sum(high_conf) == 0:
        return 0.0
    return np.mean(predictions[high_conf] != labels[high_conf])


def compute_calibration_metrics(y_true, y_prob, y_pred):
    """Compute all calibration metrics"""
    confidence = np.maximum(y_prob, 1 - y_prob)

    results = {}
    results['ECE'] = compute_ece(confidence, y_pred, y_true, n_bins=15)
    results['Brier'] = compute_brier(y_prob, y_true)
    results['NLL'] = compute_nll(y_prob, y_true)
    results['Avg_Conf'] = np.mean(confidence)
    results['Acc'] = metrics.accuracy_score(y_true, y_pred)
    results['High_Conf_Error'] = compute_high_conf_error(confidence, y_pred, y_true, threshold=0.9)
    return results


def fmt(mean_val, std_val):
    return f'{mean_val:.4f}±{std_val:.4f}'


def process_dataframe(df, method_col='method'):
    """Group by seed, setting, shot, method and compute calibration metrics per seed."""
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
    """Aggregate across seeds: mean ± std."""
    agg_cols = ['ECE', 'Brier', 'NLL', 'Avg_Conf', 'Acc', 'High_Conf_Error']
    summary = metrics_df.groupby(['setting', 'shot', 'method']).agg(
        {c: ['mean', 'std'] for c in agg_cols}
    ).reset_index()
    # Flatten columns
    summary.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c
                       for c in summary.columns]
    return summary


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))

    # Load all prediction CSVs
    dfs = {}

    # 1. PharDDIE
    phar_path = os.path.join(BASE, 'results/predictions/predictions_dataset1_PharDDIE.csv')
    if os.path.exists(phar_path):
        dfs['PharDDIE'] = pd.read_csv(phar_path)
        print(f'Loaded PharDDIE: {len(dfs["PharDDIE"])} samples')
        # ---- 种子独立性验证：校准表 PharDDIE 行必须覆盖 5 个训练种子 ----
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

    # 2. PharDDIE w/o uncertainty (VAE)
    wo_path = os.path.join(BASE, 'results/predictions/predictions_dataset1_wo_uncertainty.csv')
    if os.path.exists(wo_path):
        dfs['PharDDIE w/o VAE'] = pd.read_csv(wo_path)
        print(f'Loaded PharDDIE w/o VAE: {len(dfs["PharDDIE w/o VAE"])} samples')

    # 3. Zero-shot variants (EviDDIE)
    zs_paths = [
        os.path.join(BASE, '..', 'EviDDIE', 'results', 'predictions', 'predictions_dataset1_zero_shot_variants.csv'),
        os.path.join(BASE, 'results', 'predictions', 'predictions_dataset1_zero_shot_variants.csv'),
    ]
    for p in zs_paths:
        if os.path.exists(p):
            dfs['Zero-shot'] = pd.read_csv(p)
            print(f'Loaded Zero-shot variants: {len(dfs["Zero-shot"])} samples from {p}')
            break

    # Process all dataframes
    all_results = []
    for label, df in dfs.items():
        metrics_df = process_dataframe(df)
        all_results.append(metrics_df)

    combined = pd.concat(all_results, ignore_index=True)
    summary = aggregate_metrics(combined)

    # Also save detail CSV
    detail_csv = os.path.join(BASE, 'results/table3_complete_detail.csv')
    combined.to_csv(detail_csv, index=False, float_format='%.6f')
    print(f'\nDetail per seed saved to: {detail_csv}')

    # ============================================================
    # Print formatted table
    # ============================================================
    setting_order = ['common', 'fewer', 'rare']
    shot_order = [0, 1, 5, 10]
    # Define row order: for each setting, show all methods grouped by shot
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
    lines.append('  - ECE = Expected Calibration Error (15 equal-width bins).')
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

    # Save
    txt_path = os.path.join(BASE, 'results/table3_complete.txt')
    csv_path = os.path.join(BASE, 'results/table3_complete.csv')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f'\nSaved to: {txt_path}')

    # Also save as CSV
    summary.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'Saved to: {csv_path}')


if __name__ == '__main__':
    main()
