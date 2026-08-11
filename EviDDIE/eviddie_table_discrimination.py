#!/usr/bin/env python
# coding=utf-8
"""
EviDDIE Zero-Shot Discrimination Table.
Generates a table comparable to Table 2 but for the zero-shot setting.
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn import metrics


def compute_metrics(y_true, y_prob, group_by_event):
    y_pred = (y_prob >= 0.5).astype(int)
    r = {}
    if len(np.unique(y_true)) > 1:
        r['AUC'] = metrics.roc_auc_score(y_true, y_prob)
        p, rec, _ = metrics.precision_recall_curve(y_true, y_prob)
        r['AUPR'] = metrics.auc(rec, p)
    else:
        r['AUC'], r['AUPR'] = np.nan, np.nan
    r['ACC'] = metrics.accuracy_score(y_true, y_pred)
    event_f1s = []
    for e in np.unique(group_by_event):
        m = group_by_event == e
        if m.sum() > 0 and len(np.unique(y_true[m])) > 1:
            event_f1s.append(metrics.f1_score(y_true[m], y_pred[m], zero_division=0))
    r['Macro-F1'] = np.mean(event_f1s) if event_f1s else 0.0
    return r


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    csv_paths = [
        os.path.join(BASE, 'results/predictions/predictions_dataset1_zero_shot_variants.csv'),
        os.path.join(BASE, '..', 'PharDDIE', 'results/predictions/predictions_dataset1_zero_shot_variants.csv'),
    ]
    df = None
    for p in csv_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f'Loaded: {p}')
            break
    if df is None:
        print('FATAL: Zero-shot variants CSV not found. Run eviddie_export_zs_v2.py first.')
        sys.exit(1)

    SETTINGS = {'common': 'common (dev)', 'fewer': 'fewer (test)', 'rare': 'rare (test2)'}
    METHODS = ['Softmax baseline', 'EviDDIE w/o EVI', 'EviDDIE']

    def fmt(val, std):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '-'
        return f'{val:.4f}+/-{std:.4f}'

    seed_col = 'train_seed' if 'train_seed' in df.columns else 'seed'

    lines = []
    lines.append('=' * 120)
    lines.append('EviDDIE Zero-Shot Discrimination Performance (complement to Table 2)')
    lines.append('Shot = 0 means ZERO-SHOT: no molecular support examples for target events.')
    lines.append('=' * 120)
    H = (f"{'Setting':<18} {'Shot':<6} {'Method':<24} "
         f"{'AUC up':<22} {'AUPR up':<22} {'ACC up':<22} {'Macro-F1 up':<22}")
    lines.append(H)
    lines.append('-' * 120)

    for s_key, s_label in SETTINGS.items():
        for method in METHODS:
            sub = df[(df['setting'] == s_key) & (df['method'] == method)]
            if len(sub) == 0:
                continue
            rows = []
            for sv, g in sub.groupby(seed_col):
                if len(g) == 0:
                    continue
                m = compute_metrics(g['y_true'].values, g['prob'].values, g['event_type'].values)
                rows.append(m)
            if not rows:
                continue
            rd = pd.DataFrame(rows)
            mean_s, std_s = rd.mean(), rd.std()
            lines.append(f'{s_label:<18} {"0":<6} {method:<24} '
                         f'{fmt(mean_s.get("AUC", None), std_s.get("AUC", None)):<22} '
                         f'{fmt(mean_s.get("AUPR", None), std_s.get("AUPR", None)):<22} '
                         f'{fmt(mean_s.get("ACC", None), std_s.get("ACC", None)):<22} '
                         f'{fmt(mean_s.get("Macro-F1", None), std_s.get("Macro-F1", None)):<22}')
        lines.append('-' * 120)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - "0" shot = zero-shot setting (no support examples for target events).')
    lines.append('  - EviDDIE uses BioSentVec event prototypes instead of molecular support.')
    if seed_col == 'train_seed':
        lines.append(f'  - Mean +/- std computed across {seed_col} (training variability).')
    else:
        lines.append(f'  - Mean +/- std computed across {seed_col} (negative-sampling variability).')

    out = '\n'.join(lines)
    print(out)
    os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
    out_path = os.path.join(BASE, 'results/eviddie_discrimination_table.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()
