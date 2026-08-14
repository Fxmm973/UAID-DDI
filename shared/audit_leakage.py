#!/usr/bin/env python
# coding=utf-8
"""
Six-part leakage audit (reviewer-required). Writes one report per check under
audit/leakage_reports/ and exits non-zero if any hard check fails.

  1. support-query      : support triples vs query triples within each event/split
  2. positive-negative  : positive triples vs manifest negatives (same event)
  3. ordered-triple     : directed (d_i, e, d_j) overlap across train/dev/test/test2
  4. unordered-pair     : undirected drug-pair overlap across splits
  5. cross-split        : positive triples of one split used as negatives elsewhere
  6. kg-edge-leakage    : held-out drug pairs appearing as DRKG path_graph edges
                          (path_graph feeds the ACI neighbor encoder)

Checks 1-5 are hard failures (they should never occur). Check 6 is reported
informationally: any held-out drug pair that is also a path_graph edge is
listed, since ACI aggregates DRKG neighbours for evaluation drugs.
"""
import json
import os
import sys
import argparse

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
SPLITS = ['train_tasks', 'dev_tasks', 'test_tasks', 'test2_tasks']


def load_tasks(ds, name):
    p = os.path.join(ds, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def evt_lists(tasks):
    return {evt: [(t[0], t[1], t[2]) for t in lst] for evt, lst in tasks.items()}


def all_triples(evt):
    s = set()
    for lst in evt.values():
        s |= set(lst)
    return s


def write(path, title, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(title + '\n' + '=' * len(title) + '\n')
        f.write('\n'.join(lines) + '\n')
    print(f'  -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='PharDDIE/dataset1')
    ap.add_argument('--few', type=int, default=1, help='support size used in few-shot evaluation')
    args = ap.parse_args()
    ds = args.dataset
    out = 'audit/leakage_reports'
    few = args.few
    fail = False

    tasks = {n: load_tasks(ds, n) for n in SPLITS}
    evt = {n: evt_lists(tasks[n]) for n in SPLITS}
    allt = {n: all_triples(evt[n]) for n in SPLITS}

    # ---- 1. support-query overlap ----
    lines, ok1 = [], True
    for n in SPLITS:
        for e, lst in evt[n].items():
            sup, qry = set(lst[:few]), set(lst[few:])
            ov = sup & qry
            if ov:
                ok1 = False
                lines.append(f'[{n}] {e}: {len(ov)} overlapping triples between support and query')
    write(os.path.join(out, '01_support_query.txt'),
          'AUDIT 1: support-query overlap',
          lines if lines else ['PASS: no support-query overlap in any split.'])
    if not ok1:
        fail = True

    # ---- 2. positive-negative overlap (all seeds, dev/test/test2) ----
    # 硬失败：正样本三元组被用作负样本（泄漏）。
    # 信息性记录：同一事件内不同正样本共享同一个负样本（冗余但非泄漏）。
    lines, ok2 = [], True
    dup_total = 0
    for split in ['dev', 'test', 'test2']:
        for seed in SEEDS:
            mp = os.path.join(ds, 'neg_manifests', f'{split}_seed{seed}_negatives.json')
            if not os.path.exists(mp):
                ok2 = False
                lines.append(f'[{split} seed{seed}] manifest missing')
                continue
            m = json.load(open(mp))
            for e, entries in m.items():
                pos = set(evt[f'{split}_tasks'].get(e, []))
                neg = {(d_i, rel, d_k) for d_i, d_j, d_k, rel in entries}
                ov = pos & neg
                if ov:
                    ok2 = False
                    lines.append(f'[{split} seed{seed}] {e}: {len(ov)} positive-negative overlaps')
                if len(neg) != len(entries):
                    dup_total += len(entries) - len(neg)
                    lines.append(f'[info] [{split} seed{seed}] {e}: '
                                 f'{len(entries) - len(neg)} duplicate negatives (shared across positives; not leakage)')
    if not lines or (ok2 and dup_total == 0):
        lines = ['PASS: no positive-negative overlap, no duplicate negatives.']
    elif ok2:
        lines = ['PASS: no positive-negative overlap (hard check).'] + lines
    write(os.path.join(out, '02_positive_negative.txt'),
          'AUDIT 2: positive-negative overlap',
          lines)
    if not ok2:
        fail = True

    # ---- 3. ordered-triple cross-split ----
    lines, ok3 = [], True
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            ov = allt[SPLITS[i]] & allt[SPLITS[j]]
            if ov:
                ok3 = False
                lines.append(f'{SPLITS[i]} x {SPLITS[j]}: {len(ov)} shared directed triples')
    write(os.path.join(out, '03_ordered_triple.txt'),
          'AUDIT 3: ordered-triple cross-split overlap',
          lines if lines else ['PASS: no directed triple appears in two splits.'])
    if not ok3:
        fail = True

    # ---- 4. unordered-pair cross-split ----
    # 硬失败：同一事件下的反向无序药物对 (d_i,e,d_j)/(d_j,e,d_i) 出现在不同 split（真泄漏）。
    # 信息性记录：不同事件共享同一药物对（事件级划分下的预期现象，论文 Limitations 已说明）。
    lines, ok4 = [], True
    def unordered_event_cond(n):
        s = set()
        for e, lst in evt[n].items():
            for (a, r, b) in lst:
                s.add((e, tuple(sorted([a, b]))))
        return s
    uec = {n: unordered_event_cond(n) for n in SPLITS}
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            ov = uec[SPLITS[i]] & uec[SPLITS[j]]
            if ov:
                ok4 = False
                lines.append(f'{SPLITS[i]} x {SPLITS[j]}: {len(ov)} event-conditioned unordered pairs '
                             f'shared (reversed-triple leakage)')
    def unordered_plain(n):
        s = set()
        for lst in evt[n].values():
            for (a, r, b) in lst:
                s.add(tuple(sorted([a, b])))
        return s
    up = {n: unordered_plain(n) for n in SPLITS}
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            ov = up[SPLITS[i]] & up[SPLITS[j]]
            if ov:
                lines.append(f'[info] {SPLITS[i]} x {SPLITS[j]}: {len(ov)} unordered drug pairs '
                             f'shared across splits (different events; expected under event-level splits)')
    lines = (['PASS: no event-conditioned unordered-pair leakage across splits.'] + lines) if ok4 else lines
    write(os.path.join(out, '04_unordered_pair.txt'),
          'AUDIT 4: unordered-pair cross-split overlap',
          lines)
    if not ok4:
        fail = True

    # ---- 5. cross-split positive-negative conflicts ----
    lines, ok5 = [], True
    neg_all = set()
    for split in ['dev', 'test', 'test2']:
        mp = os.path.join(ds, 'neg_manifests', f'{split}_seed19940419_negatives.json')
        if os.path.exists(mp):
            m = json.load(open(mp))
            for e, entries in m.items():
                for d_i, d_j, d_k, rel in entries:
                    neg_all.add((d_i, rel, d_k))
    for n in SPLITS:
        ov = allt[n] & neg_all
        if ov:
            ok5 = False
            lines.append(f'{n}: {len(ov)} positive triples used as negatives elsewhere')
    write(os.path.join(out, '05_cross_split_posneg.txt'),
          'AUDIT 5: cross-split positive-negative conflicts',
          lines if lines else ['PASS: no cross-split positive-negative conflicts.'])
    if not ok5:
        fail = True

    # ---- 6. KG-edge leakage (informational report) ----
    lines, ok6 = [], True
    held = allt['test_tasks'] | allt['test2_tasks']
    held_pairs = set()
    for (a, r, b) in held:
        held_pairs.add((a, b))
        held_pairs.add((b, a))
    pg = os.path.join(ds, 'path_graph')
    if os.path.exists(pg):
        edges = set()
        with open(pg) as f:
            for line in f:
                parts = line.rstrip().split('\t')
                if len(parts) >= 3:
                    edges.add((parts[0][-7:], parts[2][-7:]))
        leaks = held_pairs & edges
        lines.append(f'held-out drug pairs checked: {len(held_pairs)}; DRKG path_graph edges: {len(edges)}')
        lines.append(f'held-out drug pairs that are also path_graph edges (ACI neighbour source): {len(leaks)}')
        for p in sorted(leaks)[:100]:
            lines.append(f'  {p[0]} -- {p[1]}')
    else:
        ok6 = False
        lines.append('path_graph not found; check skipped')
    write(os.path.join(out, '06_kg_edge_leakage.txt'),
          'AUDIT 6: KG-edge leakage (informational; see paper Limitations on DRKG coverage)',
          lines)
    if not ok6:
        fail = True

    print('LEAKAGE AUDIT:', 'FAIL' if fail else 'PASS')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
