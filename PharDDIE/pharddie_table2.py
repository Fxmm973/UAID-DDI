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
# 注: 这些 baseline 只有 1-shot 和 5-shot，无 10-shot；无 AUPR。
#     META-DDIE/RareDDIE 转录了 common/fewer/rare 三个 setting 的值；
#     其余 6 个方法只转录论文 Table 2 使用的 rare 测试值
#     （common/fewer 行输出 '—'，见文末 Notes）。
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
    'GMatching': {
        'rare': {'AUC':{1:(0.8711,0.0263),5:(0.9366,0.0212)},
                 'ACC':{1:(0.8000,0.0367),5:(0.8448,0.0289)},
                 'F1': {1:(0.8099,0.0384),5:(0.8494,0.0290)}},
    },
    'MRCGNN': {
        'rare': {'AUC':{1:(0.8528,0.0246),5:(0.9561,0.0128)},
                 'ACC':{1:(0.8122,0.0307),5:(0.8448,0.0196)},
                 'F1': {1:(0.8155,0.0304),5:(0.8610,0.0151)}},
    },
    'MetaR-Pre': {
        'rare': {'AUC':{1:(0.7938,0.0449),5:(0.7862,0.0186)},
                 'ACC':{1:(0.7162,0.0541),5:(0.6964,0.0209)},
                 'F1': {1:(0.7294,0.0616),5:(0.7082,0.0249)}},
    },
    'MetaR-In': {
        'rare': {'AUC':{1:(0.7318,0.0638),5:(0.8674,0.0332)},
                 'ACC':{1:(0.6786,0.0644),5:(0.7758,0.0329)},
                 'F1': {1:(0.6892,0.0622),5:(0.7814,0.0344)}},
    },
    'KnowDDI': {
        'rare': {'AUC':{1:(0.7282,0.0034),5:(0.9130,0.0017)},
                 'ACC':{1:(0.6183,0.0088),5:(0.8586,0.0253)},
                 'F1': {1:(0.4296,0.0209),5:(0.8343,0.0343)}},
    },
    'DSN-DDI': {
        'rare': {'AUC':{1:(0.6174,0.0653),5:(0.7176,0.0584)},
                 'ACC':{1:(0.6000,0.0537),5:(0.6483,0.0611)},
                 'F1': {1:(0.5293,0.0721),5:(0.5857,0.0758)}},
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
    if no rows match the (setting, shot) filter.

    Grouping logic:
    - If 'train_seed' column present: group by train_seed (training variability)
    - Otherwise: group by 'seed' (negative-sampling variability, legacy mode)
    """
    sub = df
    if 'setting' in df.columns:
        sub = sub[sub['setting'] == setting_key]
    if 'shot' in df.columns:
        sub = sub[sub['shot'] == shot_val]

    # 选择分组键：优先 train_seed（训练变异），其次 seed（负样本变异）
    seed_col = 'train_seed' if 'train_seed' in df.columns else 'seed'
    if seed_col not in df.columns:
        return None, None

    rows = []
    for seed_val, g in sub.groupby(seed_col):
        if len(g) == 0:
            continue
        m = compute_metrics(g['y_true'].values, g['prob'].values,
                            g['event_type'].values)
        rows.append(m)
    if not rows:
        return None, None
    rd = pd.DataFrame(rows)
    return rd.mean(), rd.std(ddof=0)  # population SD (÷5), matching the paper and aggregate_rareddie.py

def main():
    # ----------------------------------------------------------
    # 1.  Load prediction CSVs (all required – fail if missing)
    # ----------------------------------------------------------
    phar_csv = 'results/predictions/predictions_dataset1_PharDDIE.csv'
    df_phar = _require_csv(phar_csv, 'PharDDIE predictions CSV')

    # ---- 种子独立性验证：逐样本 CSV 必须覆盖 5 个训练种子 ----
    seed_col = ('train_seed' if 'train_seed' in df_phar.columns
                else ('seed' if 'seed' in df_phar.columns else None))
    if seed_col is not None:
        n_seeds = df_phar[seed_col].nunique()
        print(f'[SEED-CHAIN] PharDDIE predictions CSV: {n_seeds} distinct {seed_col} values.')
        if n_seeds != 5:
            print(f'FATAL: expected 5 training seeds in {phar_csv}, found {n_seeds}. '
                  f'Re-export with pharddie_export_full.py '
                  f'(5 independent checkpoints required; no fallback allowed).')
            sys.exit(1)

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
    lines.append('External baseline values (8 methods): transcribed from Ren et al.')
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
                lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE (Ours)":<28} '
                             f'{fmt(mean_s["AUC"], std_s["AUC"]):<22} '
                             f'{fmt(mean_s["AUPR"], std_s["AUPR"]):<22} '
                             f'{fmt(mean_s["ACC"], std_s["ACC"]):<22} '
                             f'{fmt(mean_s["Macro-F1"], std_s["Macro-F1"]):<22}')
            else:
                lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE (Ours)":<28} '
                             f'{"— (no CSV rows)":<22} {"—":<22} {"—":<22} {"—":<22}')

            # ====================================================
            # PharDDIE w/o VAE (ablation) — computed from CSV
            # ====================================================
            if has_wo:
                mean_w, std_w = _compute_rows_from_csv(df_wo, s_key, shot, ['seed'])
                if mean_w is not None:
                    lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE w/o SRAE (Ablation)":<28} '
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
            for bm in ['META-DDIE', 'RareDDIE', 'GMatching', 'MRCGNN', 'MetaR-Pre', 'MetaR-In', 'KnowDDI', 'DSN-DDI']:
                if shot not in [1, 5]:
                    lines.append(f'{s_label:<10} {shot:<6} {bm + " (Reported)":<28} '
                                 f'{"— (no 10-shot in paper)":<22} '
                                 f'{"—":<22} {"—":<22} {"—":<22}')
                else:
                    lines.append(f'{s_label:<10} {shot:<6} {bm + " (Reported)":<28} '
                                 f'{baseline_fmt(bm, s_key, "AUC", shot):<22} '
                                 f'{"—":<22} '
                                 f'{baseline_fmt(bm, s_key, "ACC", shot):<22} '
                                 f'{baseline_fmt(bm, s_key, "F1", shot):<22}')

        lines.append('-' * 135)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - "common" = validation, "fewer" = test (20-50/event), "rare" = test2 (<20/event).')
    lines.append('  - PharDDIE & ablation rows: mean +/- std computed across training seeds (if train_seed column')
    lines.append('    present) or negative-sampling seeds (legacy mode, if only seed column).')
    lines.append('  - When per-training-seed checkpoints are available, std = training variability.')
    lines.append('  - When only one checkpoint exists, std = negative-sampling variability (5 fixed manifests).')
    lines.append('  - All external baselines: transcribed from Ren et al. (Nat. Commun. 2025) source data;')
    lines.append('    none of them were re-trained or re-evaluated in this study.')
    lines.append('  - Transcribed baselines only published 1/5-shot results; no 10-shot or AUPR reported.')
    lines.append('  - META-DDIE and RareDDIE include common/fewer/rare transcribed values; the other six')
    lines.append('    methods embed only the rare-test values used in the paper Table 2 (common/fewer = \'—\').')
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
