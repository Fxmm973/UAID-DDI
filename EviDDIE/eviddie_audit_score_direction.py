#!/usr/bin/env python
# coding=utf-8
"""
P0-2 逐样本分数方向诊断（无需 GPU，读预测 CSV）。

输出：
  1) pooled 与 event-macro 的 AUROC / AUPRC / balanced accuracy（按 setting x method）
  2) 逐事件明细（样本数、正负数、mean(p|y=1) vs mean(p|y=0)、AUROC、AUROC(1-p) 诊断、
     AUPRC、balanced acc）-> results/score_direction_audit.csv
  3) 概率最小/最大的正样本与负样本示例
  4) manifest 正负配对校验（每个事件条目数、head/正尾/关系是否与 CSV 正样本一致）

判断逻辑（GPT 方案 3.3）：
  - 所有事件都 mean_pos < mean_neg -> 优先检查任务原型、manifest 对齐和事件索引
  - 只有少数事件反向 -> 模型对具体事件语义泛化失败（事件级分析）
  - pooled AUROC < 0.5 但多数 per-event AUROC > 0.5 -> 事件间概率尺度不一致，同时报告 event-macro AUROC

运行：python eviddie_audit_score_direction.py [--csv results/predictions/predictions_dataset1_zero_shot_variants.csv]
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def audit_event(y, p):
    return {
        'n': len(y),
        'n_pos': int(y.sum()),
        'mean_pos': float(p[y == 1].mean()) if (y == 1).any() else np.nan,
        'mean_neg': float(p[y == 0].mean()) if (y == 0).any() else np.nan,
        'auroc': float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        'auroc_inverse_diagnostic': float(roc_auc_score(y, 1.0 - p)) if len(np.unique(y)) > 1 else np.nan,
        'auprc': float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        'balanced_acc': float(balanced_accuracy_score(y, p >= 0.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/predictions/predictions_dataset1_zero_shot_variants.csv')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f'FATAL: CSV not found: {args.csv}. Run eviddie_export_zs_v2.py first.')
        raise SystemExit(1)

    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    print(f'Loaded {len(rows)} rows from {args.csv}')
    methods = sorted(set(r['method'] for r in rows))
    settings = sorted(set(r['setting'] for r in rows))
    print(f'methods: {methods} | settings: {settings}\n')

    # ---- 1) pooled 与 event-macro 汇总 ----
    print('=' * 110)
    print('POOLED vs EVENT-MACRO SUMMARY')
    print('=' * 110)
    for setting in settings:
        for method in methods:
            sub = [r for r in rows if r['setting'] == setting and r['method'] == method]
            if not sub:
                continue
            y = np.array([float(r['y_true']) for r in sub])
            p = np.array([float(r['prob']) for r in sub])
            ev_aucs = []
            for evt in set(r['event_type'] for r in sub):
                g = [(float(r['y_true']), float(r['prob'])) for r in sub if r['event_type'] == evt]
                yy = np.array([a for a, _ in g]); pp = np.array([b for _, b in g])
                if len(np.unique(yy)) > 1:
                    ev_aucs.append(roc_auc_score(yy, pp))
            pooled = roc_auc_score(y, p)
            macro = float(np.mean(ev_aucs)) if ev_aucs else np.nan
            print(f'{setting:8s} {method:20s} pooled AUROC={pooled:.4f} '
                  f'event-macro AUROC={macro:.4f} (n_events={len(ev_aucs)})')

    # ---- 2) 逐事件明细 ----
    print('\n' + '=' * 110)
    print('PER-EVENT AUDIT')
    print('=' * 110)
    audit_rows = []
    for setting in settings:
        for method in methods:
            sub = [r for r in rows if r['setting'] == setting and r['method'] == method]
            for evt in sorted(set(r['event_type'] for r in sub)):
                g = [r for r in sub if r['event_type'] == evt]
                y = np.array([float(r['y_true']) for r in g])
                p = np.array([float(r['prob']) for r in g])
                m = audit_event(y, p)
                audit_rows.append({'setting': setting, 'method': method,
                                   'event': evt[:60], **m})
                flag = ''
                if m['mean_pos'] < m['mean_neg']:
                    flag = '  <== REVERSED'
                print(f'{setting:8s} {method:20s} {evt[:40]:40s} '
                      f'n={m["n"]:4d} pos={m["n_pos"]:3d} '
                      f'mean_pos={m["mean_pos"]:.4f} mean_neg={m["mean_neg"]:.4f} '
                      f'AUC={m["auroc"]:.4f} AUC(1-p)={m["auroc_inverse_diagnostic"]:.4f}{flag}')
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/score_direction_audit.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    print(f'\nSaved per-event audit to {out_csv}')

    # ---- 3) 极值样本示例 ----
    print('\n' + '=' * 110)
    print('EXTREME-PROBABILITY EXAMPLES (per setting x method)')
    print('=' * 110)
    for setting in settings:
        for method in methods:
            sub = [r for r in rows if r['setting'] == setting and r['method'] == method]
            if not sub:
                continue
            pos = sorted([r for r in sub if r['y_true'] == '1'], key=lambda r: float(r['prob']))
            neg = sorted([r for r in sub if r['y_true'] == '0'], key=lambda r: -float(r['prob']))
            print(f'{setting:8s} {method:20s}: '
                  f'lowest-prob positive p={float(pos[0]["prob"]):.4f}; '
                  f'highest-prob positive p={float(pos[-1]["prob"]):.4f}; '
                  f'highest-prob negative p={float(neg[0]["prob"]):.4f}')

    # ---- 4) manifest 正负配对校验 ----
    print('\n' + '=' * 110)
    print('MANIFEST PAIRING CHECK (eval seed 19940419)')
    print('=' * 110)
    split_map = {'common': 'dev', 'fewer': 'test', 'rare': 'test2'}
    ok_all = True
    for setting in settings:
        split = split_map.get(setting)
        if not split:
            continue
        mp = f'neg_manifests/{split}_seed19940419_negatives.json'  # 与导出脚本同路径（相对 EviDDIE/）
        if not os.path.exists(mp):
            print(f'  {setting}: manifest missing ({mp})')
            continue
        manifest = json.load(open(mp, encoding='utf-8'))
        for method in methods:
            sub = [r for r in rows if r['setting'] == setting and r['method'] == method]
            bad = 0
            for evt in sorted(set(r['event_type'] for r in sub)):
                pos_rows = [r for r in sub if r['event_type'] == evt and r['y_true'] == '1']
                entries = manifest.get(evt, [])
                if len(entries) != len(pos_rows):
                    print(f'  {setting:8s} {method:20s} {evt[:40]:40s} '
                          f'COUNT MISMATCH: manifest={len(entries)} positives={len(pos_rows)}')
                    bad += 1
                    continue
                for (d_i, d_j, d_k, rel), r in zip(entries, pos_rows):
                    if not (d_i == r['drug_a'] and d_j == r['drug_b']):
                        bad += 1
            if bad == 0:
                print(f'  {setting:8s} {method:20s}: OK (all events: entry count and '
                      f'(head, positive_tail) match CSV positives)')
            else:
                ok_all = False
                print(f'  {setting:8s} {method:20s}: {bad} mismatches')
    print('\nMANIFEST PAIRING:', 'PASS' if ok_all else 'FAIL')

    # ---- 判断逻辑总结 ----
    print('\n' + '=' * 110)
    print('INTERPRETATION (GPT 方案 3.3 判断逻辑)')
    print('=' * 110)
    for setting in settings:
        for method in methods:
            ev_rows = [a for a in audit_rows if a['setting'] == setting and a['method'] == method]
            rev = sum(1 for a in ev_rows if a['mean_pos'] < a['mean_neg'])
            auc_ok = sum(1 for a in ev_rows if a['auroc'] > 0.5)
            if rev == len(ev_rows):
                verdict = 'ALL events reversed -> check prototypes/manifest alignment'
            elif rev > 0:
                verdict = f'{rev}/{len(ev_rows)} events reversed -> event-level generalization failure'
            else:
                verdict = 'no event reversed'
            print(f'{setting:8s} {method:20s}: {verdict} '
                  f'({auc_ok}/{len(ev_rows)} events with per-event AUROC>0.5)')


if __name__ == '__main__':
    main()
