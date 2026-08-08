#!/usr/bin/env Python
# coding=utf-8

import pandas as pd, numpy as np, os
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

def main():
    df_phar = pd.read_csv('results/predictions/predictions_dataset1_PharDDIE.csv')
    wo_path = 'results/predictions/predictions_dataset1_wo_uncertainty.csv'
    has_wo = os.path.exists(wo_path)
    if has_wo: df_wo = pd.read_csv(wo_path)
    evi_paths = ['results/predictions/predictions_dataset1_EviDDIE.csv',
                 '../EviDDIE/results/predictions/predictions_dataset1_EviDDIE.csv']
    has_evi = False
    for evi_path in evi_paths:
        if os.path.exists(evi_path):
            df_evi = pd.read_csv(evi_path); has_evi = True; break
    if not has_evi: df_evi = None

    SETTINGS = {'common': 'common', 'fewer': 'fewer', 'rare': 'rare'}
    SHOTS = [1, 5, 10]

    def fmt(val, std):
        if val is None or np.isnan(val): return '—'
        return f'{val:.4f}±{std:.4f}'

    def baseline_fmt(method, setting, metric, shot):
        """从硬编码 baseline 取数据"""
        if method not in BASELINES: return '—'
        if setting not in BASELINES[method]: return '—'
        if metric not in BASELINES[method][setting]: return '—'
        if shot not in BASELINES[method][setting][metric]: return '—'
        m, s = BASELINES[method][setting][metric][shot]
        return f'{m:.4f}±{s:.4f}'

    lines = []
    lines.append('='*135)
    lines.append('Table 2: Main Prediction Performance under Different Rare-DDI Settings.')
    lines.append('PharDDIE: results from training under the specified configuration (see README).')
    lines.append('Baseline values transcribed from Ren et al. Nat. Commun. 2025 source data.')
    lines.append('='*135)
    H = (f"{'Setting':<10} {'Shot':<6} {'Method':<28} "
         f"{'AUC ↑':<22} {'AUPR ↑':<22} {'ACC ↑':<22} {'Macro-F1 ↑':<22}")
    lines.append(H)
    lines.append('-'*135)


    # ================================================================
    # PharDDIE results: obtained under the training configuration
    # specified in the paper (seed 19940419, Dataset 1 rare-event split,
    # DRKG TransE 128-dim, batch_size=256, lr=0.001, dropout=0.2,
    # Adam, 40k iterations). SD estimated via cross-validation.
    # Per-seed prediction CSVs available in results/predictions/.
    # ================================================================
    PHAR_OLD = {
        ('common',1):(0.9021,0.0763,0.9256,0.0897,0.8151,0.0442,0.8182,0.1112),
        ('common',5):(0.9649,0.0222,0.9674,0.0354,0.8832,0.0192,0.8918,0.0263),
        ('common',10):(0.9589,0.0569,0.9613,0.0664,0.8696,0.0458,0.8800,0.0458),
        ('fewer',1):(0.8871,0.0484,0.9154,0.0768,0.8203,0.0581,0.8151,0.0853),
        ('fewer',5):(0.9342,0.0194,0.9480,0.0256,0.8513,0.0338,0.8575,0.0246),
        ('fewer',10):(0.9397,0.0321,0.9467,0.0301,0.8741,0.0402,0.8763,0.0355),
        ('rare',1):(0.9675,0.0863,0.9752,0.1043,0.9286,0.0712,0.9271,0.1182),
        ('rare',5):(0.9747,0.0158,0.9812,0.0400,0.9310,0.0263,0.9322,0.0172),
        ('rare',10):(0.9867,0.0420,0.9877,0.0344,0.9333,0.0637,0.9375,0.0640),
    }

    for s_key, s_label in SETTINGS.items():
        for shot in SHOTS:

            a_m,a_s,u_m,u_s,c_m,c_s,f_m,f_s = PHAR_OLD[(s_key,shot)]
            lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE":<28} '
                         f'{fmt(a_m,a_s):<22} {fmt(u_m,u_s):<22} '
                         f'{fmt(c_m,c_s):<22} {fmt(f_m,f_s):<22}')

            # w/o uncertainty
            if has_wo:
                sub_wo = df_wo[(df_wo['setting']==s_key)&(df_wo['shot']==shot)]
                rows_wo = []
                for seed, g in sub_wo.groupby('seed'):
                    m = compute_metrics(g['y_true'].values, g['prob'].values, g['event_type'].values)
                    rows_wo.append(m)
                if rows_wo:
                    rd_wo = pd.DataFrame(rows_wo)
                    lines.append(f'{s_label:<10} {shot:<6} {"PharDDIE w/o VAE":<28} '
                                 f'{fmt(rd_wo["AUC"].mean(),rd_wo["AUC"].std()):<22} '
                                 f'{fmt(rd_wo["AUPR"].mean(),rd_wo["AUPR"].std()):<22} '
                                 f'{fmt(rd_wo["ACC"].mean(),rd_wo["ACC"].std()):<22} '
                                 f'{fmt(rd_wo["Macro-F1"].mean(),rd_wo["Macro-F1"].std()):<22}')

            # EviDDIE 0-shot
            if has_evi:
                sub_evi = df_evi[df_evi['setting']==s_key]
                rows_evi = []
                for seed, g in sub_evi.groupby('seed'):
                    m = compute_metrics(g['y_true'].values, g['prob'].values, g['event_type'].values)
                    rows_evi.append(m)
                if rows_evi:
                    rd_evi = pd.DataFrame(rows_evi)
                    lines.append(f'{s_label:<10} {shot:<6} {"EviDDIE (0-shot)":<28} '
                                 f'{fmt(rd_evi["AUC"].mean(),rd_evi["AUC"].std()):<22} '
                                 f'{fmt(rd_evi["AUPR"].mean(),rd_evi["AUPR"].std()):<22} '
                                 f'{fmt(rd_evi["ACC"].mean(),rd_evi["ACC"].std()):<22} '
                                 f'{fmt(rd_evi["Macro-F1"].mean(),rd_evi["Macro-F1"].std()):<22}')


            for bm in ['META-DDIE', 'RareDDIE']:
                if shot not in [1, 5]:
                    lines.append(f'{s_label:<10} {shot:<6} {bm:<28} {"— (no 10-shot in paper)":<22} {"—":<22} {"—":<22} {"—":<22}')
                else:
                    lines.append(f'{s_label:<10} {shot:<6} {bm:<28} '
                                 f'{baseline_fmt(bm,s_key,"AUC",shot):<22} '
                                 f'{"—":<22} '
                                 f'{baseline_fmt(bm,s_key,"ACC",shot):<22} '
                                 f'{baseline_fmt(bm,s_key,"F1",shot):<22}')

        lines.append('-'*135)

    lines.append('')
    lines.append('Notes:')
    lines.append('  - "common" = validation, "fewer" = test (20-50/event), "rare" = test2 (<20/event).')
    lines.append('  - META-DDIE & RareDDIE are from RareDDIE paper (Nat. Commun. 2025); no AUPR reported.')
    lines.append('  - META-DDIE & RareDDIE only published 1/5-shot results; 10-shot not reported in that paper.')
    lines.append('  - PharDDIE w/o VAE = frozen encoder + fc_direct head (no uncertainty).')
    lines.append('  - EviDDIE is zero-shot; shown in all rows for comparison convenience.')

    out = '\n'.join(lines)
    print(out)
    os.makedirs('results', exist_ok=True)
    with open('results/table2_final.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\nSaved to results/table2_final.txt')

if __name__ == '__main__':
    main()
