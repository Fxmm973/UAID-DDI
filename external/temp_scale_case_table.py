#!/usr/bin/env python
# coding=utf-8
"""Task 18: temperature scaling for the Dataset-2 case table.

Fits ONE temperature T on the dev predictions (the T13/T18 dev split --
never on test2), applies it to the test2 predictions, and re-derives the
case-table Prob column (5-seed mean of the scaled probabilities).

Math: reused VERBATIM from the reviewed shared/calibration_table.py
(fit_temperature / apply_temperature are imported, not reimplemented):
  p        = clip(prob, 1e-7, 1-1e-7)
  z        = log(p/(1-p)) / T
  p_scaled = sigmoid(z)
  T        = argmin over T in [0.1, 10] of the binary NLL on the dev rows
             (pooled across the 5 training seeds; per-seed Ts reported too).

Outputs:
  external/outputs/case_candidates_dataset2_per_event_v2_ts.csv
      v2 case table with prob_mean replaced by prob_mean_scaled and the
      original values preserved in a new prob_mean_raw column; every other
      column and the row order are identical to the v2 table.
  external/outputs/case_candidates_dataset2_temp_scale_contrast.csv
      raw vs scaled prob for all 25 candidates.
  external/outputs/temp_scale_ds2_summary.json
      T, dev NLL before/after, test2 AUC per seed before/after.

Invariants verified (fail loudly otherwise):
  * the recomputed raw prob_mean matches the v2 table's prob_mean;
  * test2 AUC per seed is bit-identical before and after scaling (the
    scaling is a strictly monotone transform of the scores, so ranks --
    and therefore AUC -- cannot change; any change is a bug);
  * the 25-row candidate set and the row order of the v2 table are
    preserved exactly.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO)

from shared.calibration_table import (  # noqa: E402  (reviewed math, unchanged)
    apply_temperature, fit_temperature,
)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
DEV_CSV = os.path.join(OUTDIR, 'predictions_ds2_dev_0shot.csv')
TEST2_CSV = os.path.join(OUTDIR, 'predictions_ds2_retrained_0shot.csv')
CASE_CSV = os.path.join(OUTDIR, 'case_candidates_dataset2_per_event_v2.csv')
OUT_TS_CSV = os.path.join(OUTDIR, 'case_candidates_dataset2_per_event_v2_ts.csv')
OUT_CONTRAST_CSV = os.path.join(OUTDIR, 'case_candidates_dataset2_temp_scale_contrast.csv')
OUT_SUMMARY_JSON = os.path.join(OUTDIR, 'temp_scale_ds2_summary.json')

SEEDS = [19940419, 20230801, 20240520, 20260201, 20260301]


def nll_at(probs, labels, T):
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    z = np.log(p / (1 - p)) / T
    q = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(q) + (1 - labels) * np.log(1 - q)))


def per_seed_auc(df):
    return {s: float(roc_auc_score(g['y_true'].values, g['prob'].values))
            for s, g in df.groupby('train_seed')}


def main():
    dev = pd.read_csv(DEV_CSV)
    test2 = pd.read_csv(TEST2_CSV)
    case = pd.read_csv(CASE_CSV)

    # ---- 1. fit T on dev (pooled across seeds; never on test2) ----
    assert set(dev['train_seed']) == set(SEEDS), dev['train_seed'].unique()
    T = fit_temperature(dev['prob'].values, dev['y_true'].values)
    dev_nll_raw = nll_at(dev['prob'].values, dev['y_true'].values, 1.0)
    dev_nll_scaled = nll_at(dev['prob'].values, dev['y_true'].values, T)
    seed_Ts = {s: fit_temperature(g['prob'].values, g['y_true'].values)
               for s, g in dev.groupby('train_seed')}
    print(f'[T] pooled T = {T:.6f} (per-seed: ' +
          ', '.join(f'{s}:{t:.4f}' for s, t in seed_Ts.items()) + ')')
    print(f'[DEV-NLL] raw {dev_nll_raw:.6f} -> scaled {dev_nll_scaled:.6f} '
          f'(delta {dev_nll_scaled - dev_nll_raw:+.6f})')

    # ---- 2. apply to test2; AUC must be bit-identical (monotone) ----
    test2_scaled = test2.copy()
    test2_scaled['prob'] = apply_temperature(test2['prob'].values, T)
    auc_raw = per_seed_auc(test2)
    auc_scaled = per_seed_auc(test2_scaled)
    auc_diffs = {s: abs(auc_raw[s] - auc_scaled[s]) for s in SEEDS}
    assert all(d < 1e-15 for d in auc_diffs.values()), auc_diffs
    mean_sd = lambda d: (float(np.mean(list(d.values()))),  # noqa: E731
                         float(np.std(list(d.values()), ddof=1)))
    raw_mean, raw_sd = mean_sd(auc_raw)
    print(f'[AUC-TEST2] raw {raw_mean:.6f}+-{raw_sd:.6f} -> scaled '
          f'{mean_sd(auc_scaled)[0]:.6f}+-{mean_sd(auc_scaled)[1]:.6f} '
          f'(max per-seed abs diff {max(auc_diffs.values()):.2e})')

    # ---- 3. case-table Prob column: 5-seed mean, raw and scaled ----
    def prob_mean_table(df):
        g = df.groupby(['drug_a', 'drug_b', 'event_type'], as_index=False)['prob'].mean()
        return g.rename(columns={'event_type': 'event', 'prob': 'prob_mean'})

    raw_agg = prob_mean_table(test2)
    scaled_agg = prob_mean_table(test2_scaled)
    m = case.merge(raw_agg, on=['event', 'drug_a', 'drug_b'], how='left',
                   suffixes=('', '_raw'))
    m = m.merge(scaled_agg, on=['event', 'drug_a', 'drug_b'], how='left',
                suffixes=('', '_scaled'))
    assert len(m) == 25, len(m)
    assert m['prob_mean_raw'].notna().all() and m['prob_mean_scaled'].notna().all()
    diff = (m['prob_mean'] - m['prob_mean_raw']).abs()
    assert diff.max() < 1e-8, f'raw prob_mean mismatch vs v2 table: max {diff.max()}'
    print(f'[CASE] raw prob_mean reproduced (max abs diff {diff.max():.2e}); '
          f'rows {len(m)}')

    # monotonicity of the per-candidate mean under scaling (informational:
    # the CSV row order is by r, which is untouched, so this is a sanity
    # check on the mean-after-transform, not a requirement)
    v_raw = m['prob_mean'].to_numpy()
    v_scaled = m['prob_mean_scaled'].to_numpy()
    violations = sum(1 for i in range(len(v_raw)) for j in range(i + 1, len(v_raw))
                     if (v_raw[i] - v_raw[j]) * (v_scaled[i] - v_scaled[j]) < 0)
    print(f'[CASE] prob_mean pairwise order preserved under scaling: '
          f'{violations} violations of {len(v_raw) * (len(v_raw) - 1) // 2} pairs')

    # ---- 4. write outputs ----
    ts_cols = ['rank', 'event', 'drug_a', 'drug_b', 'a_name', 'b_name',
               'prob_mean', 'prob_mean_raw', 'u_mean', 'r', 'semantic_overlap',
               'evidence_auto', 'evidence_pmids', 'evidence_note']
    out_ts = m[ts_cols].copy()
    out_ts['prob_mean'] = m['prob_mean_scaled']  # Prob column = scaled value
    out_ts.to_csv(OUT_TS_CSV, index=False, encoding='utf-8-sig')  # v2 file is BOM'd
    print(f'[saved] {OUT_TS_CSV} ({len(out_ts)} rows)')

    contrast = m[['event', 'drug_a', 'drug_b', 'a_name', 'b_name',
                  'prob_mean_raw', 'prob_mean_scaled']].copy()
    contrast['delta'] = contrast['prob_mean_scaled'] - contrast['prob_mean_raw']
    contrast.to_csv(OUT_CONTRAST_CSV, index=False)
    print(f'[saved] {OUT_CONTRAST_CSV}')

    summary = {
        'T_pooled': T,
        'T_per_seed': {str(s): t for s, t in seed_Ts.items()},
        'dev_nll_raw': dev_nll_raw,
        'dev_nll_scaled': dev_nll_scaled,
        'dev_nll_delta': dev_nll_scaled - dev_nll_raw,
        'test2_auc_raw_per_seed': {str(s): v for s, v in auc_raw.items()},
        'test2_auc_scaled_per_seed': {str(s): v for s, v in auc_scaled.items()},
        'test2_auc_raw_mean': raw_mean,
        'test2_auc_raw_sd': raw_sd,
        'test2_auc_scaled_mean': mean_sd(auc_scaled)[0],
        'test2_auc_scaled_sd': mean_sd(auc_scaled)[1],
        'test2_auc_max_abs_diff': max(auc_diffs.values()),
        'test2_prob_raw_min': float(test2['prob'].min()),
        'test2_prob_raw_max': float(test2['prob'].max()),
        'test2_prob_scaled_min': float(test2_scaled['prob'].min()),
        'test2_prob_scaled_max': float(test2_scaled['prob'].max()),
        'n_dev_rows': int(len(dev)),
        'n_test2_rows': int(len(test2)),
        'n_case_rows': int(len(m)),
        'prob_mean_order_violations': violations,
    }
    with open(OUT_SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[saved] {OUT_SUMMARY_JSON}')


if __name__ == '__main__':
    main()
