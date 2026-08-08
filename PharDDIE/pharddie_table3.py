#!/usr/bin/env Python
# coding=utf-8
"""
Table 3: Calibration Performance.
从 PharDDIE CSV 和 EviDDIE CSV 计算 ECE/Brier/NLL/HC_Err。
"""
import pandas as pd, numpy as np, os
from sklearn import metrics

def compute_ece(probs, labels, n_bins=15):
    """ECE: per-bin |accuracy - confidence|, weighted by bin size."""
    preds = (probs >= 0.5).astype(int)
    conf = np.maximum(probs, 1 - probs)
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > boundaries[i]) & (conf <= boundaries[i + 1])
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
            avg_conf = conf[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(acc - avg_conf)
    return ece

def compute_cal_metrics(y_true, y_prob):
    conf = np.maximum(y_prob, 1 - y_prob)
    ece = compute_ece(y_prob, y_true, 15)
    brier = np.mean((y_prob - y_true)**2)
    nll = -np.mean(y_true*np.log(np.clip(y_prob,1e-15,1)) + (1-y_true)*np.log(np.clip(1-y_prob,1e-15,1)))
    hc_mask = conf > 0.9
    hc_err = np.mean((y_prob[hc_mask]>=0.5).astype(int) != y_true[hc_mask]) if hc_mask.sum()>0 else 0.0
    avg_conf = conf.mean()
    return {'ECE':ece, 'Brier':brier, 'NLL':nll, 'HC_Err':hc_err, 'Avg_Conf':avg_conf}

def fmt(v, s):
    return f'{v:.4f}±{s:.4f}'

def main():
    df_phar = pd.read_csv('results/predictions/predictions_dataset1_PharDDIE.csv')
    wo_path = 'results/predictions/predictions_dataset1_wo_uncertainty.csv'
    has_wo = os.path.exists(wo_path)
    if has_wo: df_wo = pd.read_csv(wo_path)
    zs_paths = ['../EviDDIE/results/predictions/predictions_dataset1_zero_shot_variants.csv',
                'results/predictions/predictions_dataset1_zero_shot_variants.csv']
    has_zs = False
    for p in zs_paths:
        if os.path.exists(p):
            df_zs = pd.read_csv(p); has_zs = True; break
    if not has_zs: df_zs = None

    lines = []
    lines.append('='*105)
    lines.append('Table 3: Calibration Performance of Deterministic and Evidential Prediction Heads.')
    lines.append('Lower is better for ECE, Brier, NLL, HC_Err.')
    lines.append('='*105)
    H = f"{'Setting':<10} {'Shot':<6} {'Method':<30} {'ECE ↓':<16} {'Brier ↓':<16} {'NLL ↓':<16} {'HC_Err ↓':<16}"
    lines.append(H)
    lines.append('-'*105)

    for s_key, s_label in [('common','common'),('fewer','fewer'),('rare','rare')]:
        for shot in [1,5,10]:
            # PharDDIE
            sub = df_phar[(df_phar['setting']==s_key)&(df_phar['shot']==shot)]
            vals = {}
            for seed, g in sub.groupby('seed'):
                m = compute_cal_metrics(g['y_true'].values, g['prob'].values)
                for k,v in m.items(): vals.setdefault(k,[]).append(v)
            rd = {k:(np.mean(v),np.std(v)) for k,v in vals.items()}
            lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE":<30} '
                         f'{fmt(*rd["ECE"]):<16} {fmt(*rd["Brier"]):<16} '
                         f'{fmt(*rd["NLL"]):<16} {fmt(*rd["HC_Err"]):<16}')

            # w/o uncertainty
            if has_wo:
                sub_wo = df_wo[(df_wo['setting']==s_key)&(df_wo['shot']==shot)]
                vals_wo = {}
                for seed, g in sub_wo.groupby('seed'):
                    m = compute_cal_metrics(g['y_true'].values, g['prob'].values)
                    for k,v in m.items(): vals_wo.setdefault(k,[]).append(v)
                if vals_wo:
                    rd_wo = {k:(np.mean(v),np.std(v)) for k,v in vals_wo.items()}
                    lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE w/o VAE":<30} '
                                 f'{fmt(*rd_wo["ECE"]):<16} {fmt(*rd_wo["Brier"]):<16} '
                                 f'{fmt(*rd_wo["NLL"]):<16} {fmt(*rd_wo["HC_Err"]):<16}')

        # EviDDIE zero-shot variants
        if has_zs:
            for zs_method in ['Softmax baseline', 'EviDDIE w/o EVI', 'EviDDIE']:
                sub_zs = df_zs[(df_zs['setting']==s_key)&(df_zs['method']==zs_method)]
                vals_zs = {}
                for seed, g in sub_zs.groupby('seed'):
                    m = compute_cal_metrics(g['y_true'].values, g['prob'].values)
                    for k,v in m.items(): vals_zs.setdefault(k,[]).append(v)
                if vals_zs:
                    rd_zs = {k:(np.mean(v),np.std(v)) for k,v in vals_zs.items()}
                    lines.append(f'{s_label:<10} {"0":<6} {zs_method:<30} '
                                 f'{fmt(*rd_zs["ECE"]):<16} {fmt(*rd_zs["Brier"]):<16} '
                                 f'{fmt(*rd_zs["NLL"]):<16} {fmt(*rd_zs["HC_Err"]):<16}')

        lines.append('-'*105)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - ECE computed with 15 equal-width bins.')
    lines.append('  - HC_Err = error rate among predictions with confidence > 0.9.')
    lines.append('  - PharDDIE w/o VAE uses deterministic fc_direct head (no uncertainty quantification).')
    lines.append('  - EviDDIE uses Dirichlet-based evidential uncertainty (EDL).')

    out = '\n'.join(lines)
    print(out)
    os.makedirs('results', exist_ok=True)
    with open('results/table3_final.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\nSaved to results/table3_final.txt')

if __name__ == '__main__':
    main()
