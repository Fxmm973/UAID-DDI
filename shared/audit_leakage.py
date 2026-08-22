#!/usr/bin/env python
# coding=utf-8
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
    ap.add_argument('--episode-manifests', default=None,
                    help='P0-5：真实评估 episode manifest 目录（导出脚本生成）。存在时审计实际 '
                         'support/query/neg 交叉，替代静态前 K 切分。')
    args = ap.parse_args()
    ds = args.dataset
    out = 'audit/leakage_reports'
    few = args.few
    fail = False

    tasks = {n: load_tasks(ds, n) for n in SPLITS}
    evt = {n: evt_lists(tasks[n]) for n in SPLITS}
    allt = {n: all_triples(evt[n]) for n in SPLITS}

    lines, ok1 = [], True
    em_dir = args.episode_manifests
    if em_dir and os.path.isdir(em_dir):
        audited = 0
        for fn in sorted(os.listdir(em_dir)):
            if not fn.endswith('.json'):
                continue
            payload = json.load(open(os.path.join(em_dir, fn), encoding='utf-8'))
            for key, ep in payload.get('episodes', {}).items():
                sup = {tuple(x) for x in ep.get('support', [])}
                qpos = {tuple(x) for x in ep.get('query_positives', [])}
                qneg = {tuple(x) for x in ep.get('query_negatives', [])}
                if sup & qpos:
                    ok1 = False
                    lines.append(f'{fn} {key}: {len(sup & qpos)} support-query overlaps')
                if qpos & qneg:
                    ok1 = False
                    lines.append(f'{fn} {key}: {len(qpos & qneg)} positive-negative overlaps')
                audited += 1
        if lines:
            lines.insert(0, f'Audited {audited} real evaluation episodes from {em_dir}.')
        else:
            lines = [f'PASS: audited {audited} real evaluation episodes from {em_dir}; '
                     f'no support-query or positive-negative overlap.']
    else:
        for n in SPLITS:
            for e, lst in evt[n].items():
                sup, qry = set(lst[:few]), set(lst[few:])
                ov = sup & qry
                if ov:
                    ok1 = False
                    lines.append(f'[{n}] {e}: {len(ov)} overlapping triples between support and query')
        if lines:
            lines.insert(0, 'NOTE: --episode-manifests not provided; static first-K check only '
                            '(runtime assertions cover training episodes).')
        else:
            lines = ['PASS: no static support-query overlap in any split. '
                     '(NOTE: --episode-manifests not provided; runtime assertions cover training episodes.)']
    write(os.path.join(out, '01_support_query.txt'),
          'AUDIT 1: support-query overlap',
          lines)
    if not ok1:
        fail = True

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

    lines, ok6 = [], True
    held = allt['test_tasks'] | allt['test2_tasks']
    held_pairs = set()
    for (a, r, b) in held:
        held_pairs.add((a, b))
        held_pairs.add((b, a))
    pg = os.path.join(ds, 'path_graph_train_only')
    if not os.path.exists(pg):
        pg = os.path.join(ds, 'path_graph')
    if os.path.exists(pg):
        edges = set()
        with open(pg) as f:
            for line in f:
                parts = line.rstrip().split('\t')
                if len(parts) >= 3:
                    edges.add((parts[0][-7:], parts[2][-7:]))
        leaks = held_pairs & edges
        lines.append(f'held-out drug pairs checked: {len(held_pairs)}; graph edges: {len(edges)} (source: {pg})')
        lines.append(f'held-out drug pairs that are also graph edges (ACI neighbour source): {len(leaks)}')
        for p in sorted(leaks)[:100]:
            lines.append(f'  {p[0]} -- {p[1]}')
        if leaks:
            ok6 = False
            lines.append('HARD FAIL: held-out drug pairs appear as ACI neighbour edges. '
                         'Run shared/build_sanitized_path_graph.py and rebuild the ACI index.')
        else:
            lines.append('PASS: 0 held-out drug-pair edges in the ACI neighbour graph.')
    else:
        ok6 = False
        lines.append('graph file not found; check skipped')
    sm = 'audit/sanitized_graph_manifest.json'
    if os.path.exists(sm):
        import json as _json
        m = _json.load(open(sm, encoding='utf-8'))
        if 'graphs' in m:  # multi-dataset manifest
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ds_key = os.path.normpath(os.path.relpath(os.path.abspath(ds), repo)).replace(os.sep, '/')
            meta = m.get('datasets', {}).get(ds_key, {})
            ghash = m.get('graphs', {}).get(f'{ds_key}/path_graph_train_only', '')
            lines.append(f'sanitized graph manifest: original={meta.get("original_edges")} '
                         f'removed={meta.get("removed_edges")} kept={meta.get("kept_edges")} '
                         f'sha256={ghash[:16]}...')
        else:  # legacy single-dataset manifest
            lines.append(f'sanitized graph manifest: original={m.get("original_edges")} '
                         f'removed={m.get("removed_edges")} kept={m.get("kept_edges")} '
                         f'sha256={m.get("path_graph_train_only_sha256", "")[:16]}...')
    write(os.path.join(out, '06_kg_edge_leakage.txt'),
          'AUDIT 6: KG-edge leakage (HARD check; ACI reads the sanitized path_graph_train_only)',
          lines)
    if not ok6:
        fail = True

    print('LEAKAGE AUDIT:', 'FAIL' if fail else 'PASS')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
