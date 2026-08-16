#!/usr/bin/env python
# coding=utf-8
"""
run_rareddie_seed.py — 单种子窗口版：一个窗口跑一个种子
每个窗口内串行：1-shot 训练(40k)→评估 → 5-shot 训练(40k)→评估
结果写入 results/rareddie_seed_{seed}.txt（五个窗口互不冲突）
用法（五个窗口分别运行）：
  python run_rareddie_seed.py --seed 19940419
  python run_rareddie_seed.py --seed 20230801
  python run_rareddie_seed.py --seed 20240115
  python run_rareddie_seed.py --seed 20240520
  python run_rareddie_seed.py --seed 20240910
跑完五个窗口后：python aggregate_rareddie.py 输出 mean±std 汇总
"""
import argparse
import os
import re
import subprocess


def train(few, seed):
    prefix = f'rareddie_{few}shot_seed{seed}'
    ckpt = f'models/{prefix}bestmodel'
    if os.path.exists(ckpt):
        print(f'[SKIP TRAIN] {ckpt} 已存在，跳过训练')
        return
    cmd = ['python', 'trainer_structure_acc_fp_neigh_VAE_struc.py',
           '--dataset', 'dataset1', '--few', str(few), '--train_few', str(few),
           '--batch_size', '256', '--max_batches', '40000',
           '--seed', str(seed), '--prefix', prefix]
    print(f'[TRAIN] {" ".join(cmd)}')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f'训练失败: {prefix} (exit {r.returncode})')
    if not os.path.exists(ckpt):
        raise SystemExit(f'训练结束但 checkpoint 不存在: {ckpt}')


def evaluate(few, seed, mode):
    ckpt = f'models/rareddie_{few}shot_seed{seed}bestmodel'
    r = subprocess.run(
        ['python', 'eval_rareddie_unified.py', '--few', str(few), '--mode', mode,
         '--seed', str(seed), '--checkpoint', ckpt],
        capture_output=True, text=True)
    out = (r.stdout or '') + (r.stderr or '')
    m = re.search(r'\[UNIFIED \w+\] AUC=([\d.]+)\s+ACC=([\d.]+)\s+macro-F1=([\d.]+)', out)
    if not m:
        print(out[-1500:])
        raise SystemExit(f'评估失败: {few}shot seed{seed} {mode}')
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    seed = args.seed

    lines = [f'RareDDIE unified eval - seed {seed}', '=' * 60]
    for few in (1, 5):
        train(few, seed)
        for mode in ('test', 'test2'):
            auc, acc, f1 = evaluate(few, seed, mode)
            lines.append(f'{few}-shot {mode:5s}: AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f}')
            print(f'[RESULT {seed}] {few}-shot {mode}: AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f}', flush=True)

    os.makedirs('results', exist_ok=True)
    with open(f'results/rareddie_seed_{seed}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[DONE {seed}] -> results/rareddie_seed_{seed}.txt')


if __name__ == '__main__':
    main()
