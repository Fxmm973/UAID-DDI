#!/usr/bin/env python
# coding=utf-8
import argparse
import os
import re
import subprocess

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
SHOTS = [1, 5]
MODES = [('test', 'fewer'), ('test2', 'rare')]
MAX_BATCHES = 40000

PUBLISHED = {
    ('test', 1): (0.8655, 0.7726, 0.7735),
    ('test', 5): (0.9351, 0.8542, 0.8560),
    ('test2', 1): (0.9392, 0.8408, 0.8507),
    ('test2', 5): (0.9879, 0.9328, 0.9370),
}


def train(few, seed, force):
    prefix = f'rareddie_{few}shot_seed{seed}'
    ckpt = f'models/{prefix}bestmodel'
    if os.path.exists(ckpt) and not force:
        print(f'[SKIP TRAIN] {ckpt} 已存在，跳过训练')
        return
    if os.path.exists(ckpt) and force:
        os.remove(ckpt)
        print(f'[FORCE] 已删除旧 checkpoint: {ckpt}')
    cmd = ['python', 'trainer_structure_acc_fp_neigh_VAE_struc.py',
           '--dataset', 'dataset1', '--few', str(few), '--train_few', str(few),
           '--batch_size', '256', '--max_batches', str(MAX_BATCHES),
           '--seed', str(seed), '--prefix', prefix]
    print(f'[TRAIN] {" ".join(cmd)}')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f'训练失败: {prefix} (exit {r.returncode})')
    if not os.path.exists(ckpt):
        raise RuntimeError(f'训练结束但 checkpoint 不存在: {ckpt}')


def evaluate(few, seed, mode):
    ckpt = f'models/rareddie_{few}shot_seed{seed}bestmodel'
    cmd = ['python', 'eval_rareddie_unified.py', '--few', str(few), '--mode', mode,
           '--seed', str(seed), '--checkpoint', ckpt]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or '') + (r.stderr or '')
    m = re.search(r'\[UNIFIED \w+\] AUC=([\d.]+)\s+ACC=([\d.]+)\s+macro-F1=([\d.]+)', out)
    if not m:
        print(out[-2000:])
        raise RuntimeError(f'评估失败: {few}shot seed{seed} {mode}')
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--few', type=int, default=0, help='0=全部; 1/5=只跑该 shot')
    ap.add_argument('--force', action='store_true', help='删除已有 checkpoint 重训')
    args = ap.parse_args()
    shots = [args.few] if args.few in SHOTS else SHOTS

    results = {s: {m: [] for m, _ in MODES} for s in shots}
    for few in shots:
        for seed in SEEDS:
            train(few, seed, args.force)
            for mode, _ in MODES:
                auc, acc, f1 = evaluate(few, seed, mode)
                results[few][mode].append((seed, auc, acc, f1))
                print(f'[RESULT] {few}shot {mode} seed{seed}: AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f}')

    os.makedirs('results', exist_ok=True)
    lines = []
    lines.append('=' * 100)
    lines.append('RareDDIE 统一协议重评结果（固定 manifest seed 19940419；训练 5 种子 × 40k 批次）')
    lines.append('=' * 100)
    lines.append('')
    for few in shots:
        for mode, label in MODES:
            rows = results[few][mode]
            n = len(rows)
            for seed, auc, acc, f1 in rows:
                lines.append(f'  {few}-shot {label:5s} seed {seed}: AUC={auc:.4f}  ACC={acc:.4f}  F1={f1:.4f}')
            lines.append('')
    lines.append('-' * 100)
    lines.append('汇总（mean ± std across seeds）:')
    lines.append('-' * 100)
    for few in shots:
        for mode, label in MODES:
            rows = results[few][mode]
            import statistics
            def ms(i):
                vals = [r[i] for r in rows]
                return statistics.fmean(vals), statistics.pstdev(vals)
            a, s_a = ms(1); c, s_c = ms(2); f, s_f = ms(3)
            pub = PUBLISHED[(mode, few)]
            lines.append(f'{few}-shot {label:5s}: AUC {a:.4f}±{s_a:.4f} | ACC {c:.4f}±{s_c:.4f} | F1 {f:.4f}±{s_f:.4f}   (论文发表值 AUC {pub[0]} ACC {pub[1]} F1 {pub[2]})')
    lines.append('')
    lines.append('注：± 为 5 种子总体标准差（÷5），与论文 Table 2 转录口径一致；')
    lines.append('    published 值为 RareDDIE 论文 Fig.3a source data，仅作参照。')
    out = '\n'.join(lines)
    with open('results/rareddie_unified_results.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print(out)
    print('\n结果已写入 results/rareddie_unified_results.txt')


if __name__ == '__main__':
    main()
