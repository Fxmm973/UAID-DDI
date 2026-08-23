#!/usr/bin/env python
# coding=utf-8
import os
import re
import statistics

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
PUBLISHED = {
    ('test', 1): (0.8655, 0.7726, 0.7735),
    ('test', 5): (0.9351, 0.8542, 0.8560),
    ('test2', 1): (0.9392, 0.8408, 0.8507),
    ('test2', 5): (0.9879, 0.9328, 0.9370),
}

rows = {(f, m): [] for f in (1, 5) for m in ('test', 'test2')}
missing = []
for seed in SEEDS:
    p = f'results/rareddie_seed_{seed}.txt'
    if not os.path.exists(p):
        missing.append(p)
        print(f'[WAIT] 缺少 {p} —— 等该窗口跑完再汇总')
        continue
    for line in open(p, encoding='utf-8'):
        m = re.match(r'(\d)-shot (test2?)\s*: AUC=([\d.]+) ACC=([\d.]+) F1=([\d.]+)', line)
        if m:
            few, mode = int(m.group(1)), m.group(2)
            rows[(few, mode)].append((float(m.group(3)), float(m.group(4)), float(m.group(5))))

out = []
out.append('=' * 96)
out.append('RareDDIE 统一协议重评汇总（固定 manifest seed 19940419；5 种子 × 40k 批次）')
out.append('=' * 96)
for few in (1, 5):
    for mode in ('test', 'test2'):
        vals = rows[(few, mode)]
        if len(vals) != 5:
            out.append(f'{few}-shot {mode:5s}: 数据不全 ({len(vals)}/5 个种子)，暂不汇总')
            continue

        def ms(i):
            v = [r[i] for r in vals]
            return statistics.fmean(v), statistics.pstdev(v)
        a, sa = ms(0)
        c, sc = ms(1)
        f, sf = ms(2)
        pub = PUBLISHED[(mode, few)]
        out.append(f'{few}-shot {mode:5s}: AUC {a:.4f}±{sa:.4f} | ACC {c:.4f}±{sc:.4f} | F1 {f:.4f}±{sf:.4f}'
                   f'   (发表值 AUC {pub[0]} ACC {pub[1]} F1 {pub[2]})')
out.append('')
out.append('注：± 为 5 种子总体标准差（÷5）；发表值来自 RareDDIE 论文 Fig.3a source data，仅作参照。')
text = '\n'.join(out)
print(text)
if not missing:
    with open('results/rareddie_unified_results.txt', 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n已写入 results/rareddie_unified_results.txt')

    tex = []
    tex.append('')
    tex.append('===== 论文 Table 2 行格式（可直接粘贴）=====')
    for label, mode in (('fewer (test)', 'test'), ('rare (test2)', 'test2')):
        a = rows[(1, mode)]; b = rows[(5, mode)]
        if len(a) != 5 or len(b) != 5:
            tex.append(f'{label}: 数据不全，暂不输出')
            continue
        def ms(i, vals):
            v = [r[i] for r in vals]
            return statistics.fmean(v), statistics.pstdev(v)
        cells = []
        for i in (0, 1, 2):
            m, s = ms(i, a)
            cells.append(f'{m:.4f}$\\pm${s:.4f}')
        for i in (0, 1, 2):
            m, s = ms(i, b)
            cells.append(f'{m:.4f}$\\pm${s:.4f}')
        tex.append(f'RareDDIE (re-evaluated) & ' + ' & '.join(cells) + f' \\\\   % {label}')
    print('\n'.join(tex))
    with open('results/rareddie_unified_results.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print('\nTable 2 格式行已追加写入 results/rareddie_unified_results.txt')
