#!/usr/bin/env Python
# coding=utf-8

import pandas as pd, numpy as np, os, sys
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

# ============================================================
# BASELINES: transcribed from Ren et al. (Nat. Commun., 2025)
# published source data (Excel 41467_2025_59431_MOESM8_ESM.xlsx,
# Sheet fig.3a).  These values were NOT regenerated in this study;
# the comparison is an external reference.
#
# 格式: BASELINES[method][setting][metric][shot] = (mean, std)
# 注: 这些 baseline 只有 1-shot 和 5-shot，无 10-shot；无 AUPR
# ============================================================
BASELINES = {
    'META-DDIE': {
        'common': {'AUC':{1:(0.5298,0.0435),5:(0.5218,0.0427)},
                   'ACC':{1:(0.5181,0.0281),5:(0.5111,0.0209)},
                   'F1': {1:(0.4251,0.0729),5:(0.3878,0.0878)}},
        'fewer':  {'AUC':{1:(0.5481,0.0644),5:(0.5432,0.0185)},
                   'ACC':{1:(0.5283,0.0313),5:(0.5216,0.0094)},
                   'F1': {1:(0.4731,0.0996),5:(0.3715,0.0987)}},
        'rare':   {'AUC':{1:(0.5482,0.0631),5:(0.5401,0.0560)},
                   'ACC':{1:(0.5255,0.0324),5:(0.5348,0.0491)},
                   'F1': {1:(0.5170,0.0725),5:(0.4679,0.0478)}},
    },
    'RareDDIE': {
        'common': {'AUC':{1:(0.8492,0.0108),5:(0.9105,0.0076)},
                   'ACC':{1:(0.7681,0.0104),5:(0.8332,0.0076)},
                   'F1': {1:(0.7759,0.0079),5:(0.8392,0.0074)}},
        'fewer':  {'AUC':{1:(0.8655,0.0119),5:(0.9351,0.0050)},
                   'ACC':{1:(0.7726,0.0145),5:(0.8542,0.0074)},
                   'F1': {1:(0.7736,0.0048),5:(0.8560,0.0090)}},
        'rare':   {'AUC':{1:(0.9392,0.0273),5:(0.9879,0.0096)},
                   'ACC':{1:(0.8408,0.0202),5:(0.9328,0.0234)},
                   'F1': {1:(0.8507,0.0186),5:(0.9370,0.0206)}},
    },
}

# ============================================================
# Helper: load a required CSV or die with a clear message
# ============================================================
def _require_csv(path, label):
    if not os.path.exists(path):
        print(f"FATAL: {label} not found at '{path}'.")
        print("Generate it first with the corresponding export script (see README).")
        sys.exit(1)
    return pd.read_csv(path)

def _compute_rows_from_csv(df, setting_key, shot_val, group_cols):
    """Return (mean_series, std_series) of metrics across seeds, or (None, None)
    if no rows match the (setting, shot) filter."""
    sub = df
    if 'setting' in df.columns:
        sub = sub[sub['setting'] == setting_key]
    if 'shot' in df.columns:
        sub = sub[sub['shot'] == shot_val]
    rows = []
    for seed, g in sub.groupby('seed'):
        m = compute_metrics(g['y_true'].values, g['prob'].values,
                            g['event_type'].values)
        rows.append(m)
    if not rows:
        return None, None
    rd = pd.DataFrame(rows)
    return rd.mean(), rd.std()

def main():
    # ----------------------------------------------------------
    # 1.  Load prediction CSVs (all required – fail if missing)
    # ----------------------------------------------------------
    phar_csv = 'results/predictions/predictions_dataset1_PharDDIE.csv'
    df_phar = _require_csv(phar_csv, 'PharDDIE predictions CSV')

    wo_path = 'results/predictions/predictions_dataset1_wo_uncertainty.csv'
    has_wo = os.path.exists(wo_path)
    if has_wo:
        df_wo = pd.read_csv(wo_path)

    evi_paths = ['results/predictions/predictions_dataset1_EviDDIE.csv',
                 '../EviDDIE/results/predictions/predictions_dataset1_EviDDIE.csv']
    has_evi = False
    for evi_path in evi_paths:
        if os.path.exists(evi_path):
            df_evi = pd.read_csv(evi_path)
            has_evi = True
            break
    if not has_evi:
        df_evi = None

    SETTINGS = {'common': 'common', 'fewer': 'fewer', 'rare': 'rare'}
    SHOTS = [1, 5, 10]

    def fmt(val, std):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '—'
        return f'{val:.4f}±{std:.4f}'

    def baseline_fmt(method, setting, metric, shot):
        """Look up hardcoded external-baseline value (transcribed from source)."""
        if method not in BASELINES:
            return '—'
        if setting not in BASELINES[method]:
            return '—'
        if metric not in BASELINES[method][setting]:
            return '—'
        if shot not in BASELINES[method][setting][metric]:
            return '—'
        m, s = BASELINES[method][setting][metric][shot]
        return f'{m:.4f}±{s:.4f}'

    lines = []
    lines.append('=' * 135)
    lines.append('Table 2: Main Prediction Performance under Different Rare-DDI Settings.')
    lines.append('PharDDIE & ablation rows: mean ± std computed from per-seed prediction CSVs.')
    lines.append('Baseline values (META-DDIE, RareDDIE): transcribed from Ren et al.')
    lines.append('  Nat. Commun. 2025 source data — NOT regenerated in this study.')
    lines.append('=' * 135)
    H = (f"{'Setting':<10} {'Shot':<6} {'Method':<28} "
         f"{'AUC ↑':<22} {'AUPR ↑':<22} {'ACC ↑':<22} {'Macro-F1 ↑':<22}")
    lines.append(H)
    lines.append('-' * 135)

    for s_key, s_label in SETTINGS.items():
        for shot in SHOTS:
            # ====================================================
            # PharDDIE (main) — computed from per-seed CSV
            # ====================================================
            mean_s, std_s = _compute_rows_from_csv(df_phar, s_key, shot, ['seed'])
            if mean_s is not None:
                lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE":<28} '
                             f'{fmt(mean_s["AUC"], std_s["AUC"]):<22} '
                             f'{fmt(mean_s["AUPR"], std_s["AUPR"]):<22} '
                             f'{fmt(mean_s["ACC"], std_s["ACC"]):<22} '
                             f'{fmt(mean_s["Macro-F1"], std_s["Macro-F1"]):<22}')
            else:
                lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE":<28} '
                             f'{"— (no CSV rows)":<22} {"—":<22} {"—":<22} {"—":<22}')

            # ====================================================
            # PharDDIE w/o VAE (ablation) — computed from CSV
            # ====================================================
            if has_wo:
                mean_w, std_w = _compute_rows_from_csv(df_wo, s_key, shot, ['seed'])
                if mean_w is not None:
                    lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE w/o SRAE":<28} '
                                 f'{fmt(mean_w["AUC"], std_w["AUC"]):<22} '
                                 f'{fmt(mean_w["AUPR"], std_w["AUPR"]):<22} '
                                 f'{fmt(mean_w["ACC"], std_w["ACC"]):<22} '
                                 f'{fmt(mean_w["Macro-F1"], std_w["Macro-F1"]):<22}')

            # ====================================================
            # EviDDIE (0-shot) — computed from CSV
            # ====================================================
            if has_evi:
                mean_e, std_e = _compute_rows_from_csv(df_evi, s_key, shot, ['seed'])
                if mean_e is not None:
                    lines.append(f'{s_label:<10} {shot:<6} {"EviDDIE (0-shot)":<28} '
                                 f'{fmt(mean_e["AUC"], std_e["AUC"]):<22} '
                                 f'{fmt(mean_e["AUPR"], std_e["AUPR"]):<22} '
                                 f'{fmt(mean_e["ACC"], std_e["ACC"]):<22} '
                                 f'{fmt(mean_e["Macro-F1"], std_e["Macro-F1"]):<22}')

            # ====================================================
            # External baselines (transcribed, not re-trained)
            # ====================================================
            for bm in ['META-DDIE', 'RareDDIE']:
                if shot not in [1, 5]:
                    lines.append(f'{s_label:<10} {shot:<6} {bm:<28} '
                                 f'{"— (no 10-shot in paper)":<22} '
                                 f'{"—":<22} {"—":<22} {"—":<22}')
                else:
                    lines.append(f'{s_label:<10} {shot:<6} {bm:<28} '
                                 f'{baseline_fmt(bm, s_key, "AUC", shot):<22} '
                                 f'{"—":<22} '
                                 f'{baseline_fmt(bm, s_key, "ACC", shot):<22} '
                                 f'{baseline_fmt(bm, s_key, "F1", shot):<22}')

        lines.append('-' * 135)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - "common" = validation, "fewer" = test (20-50/event), "rare" = test2 (<20/event).')
    lines.append('  - PharDDIE & ablation rows: mean ± std computed from per-seed prediction CSVs.')
    lines.append('  - META-DDIE & RareDDIE: transcribed from Ren et al. (Nat. Commun. 2025) source data;')
    lines.append('    these baselines were NOT re-trained or re-evaluated in this study.')
    lines.append('  - META-DDIE & RareDDIE only published 1/5-shot results; no 10-shot or AUPR reported.')
    lines.append('  - "PharDDIE w/o SRAE" = frozen encoder + fc_direct head (no uncertainty).')
    lines.append('  - EviDDIE is zero-shot; shown in all rows for comparison convenience.')
    lines.append('  - If any PharDDIE/ablation row shows "— (no CSV rows)", the corresponding')
    lines.append('    prediction CSV is missing or does not cover that (setting, shot) combination.')

    out = '\n'.join(lines)
    print(out)
    os.makedirs('results', exist_ok=True)
    with open('results/table2_final.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\nSaved to results/table2_final.txt')

if __name__ == '__main__':
    main()
