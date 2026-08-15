#!/usr/bin/env python
# coding=utf-8
"""
P0-5 (GPT 6.2)：构建净化后的 DRKG path_graph。

移除两端均为 held-out（test/test2）药物对的所有直接边（防止 ACI 邻居直接泄露
held-out 药物对关系），保留药物到基因/蛋白/疾病等非药物对上下文边。

输出：
  - {dataset}/path_graph_train_only          净化后的图（ACI 应只读取此文件）
  - audit/removed_heldout_edges.json        被移除的边清单
  - audit/sanitized_graph_manifest.json     {原始边数, 保留边数, 移除边数, 两个文件的 SHA256}

运行：python shared/build_sanitized_path_graph.py --dataset PharDDIE/dataset1
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def normalized_pair(a, b):
    return tuple(sorted((a, b)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='PharDDIE/dataset1')
    args = ap.parse_args()
    ds = Path(args.dataset)

    heldout_pairs = set()
    for split in ['test', 'test2']:
        tasks = json.load(open(ds / f'{split}_tasks.json', encoding='utf-8'))
        for event, rows in tasks.items():
            for head, relation, tail in rows:
                heldout_pairs.add(normalized_pair(head, tail))
    print(f'held-out drug pairs: {len(heldout_pairs)}')

    path_graph = ds / 'path_graph'
    removed, kept = [], []
    for line in open(path_graph, encoding='utf-8'):
        parts = line.rstrip().split('\t')
        if len(parts) < 3:
            kept.append(line)
            continue
        head, relation, tail = parts[0], parts[1], parts[2]
        if normalized_pair(head[-7:], tail[-7:]) in heldout_pairs:
            removed.append([head, relation, tail])
        else:
            kept.append(line)

    out_path = ds / 'path_graph_train_only'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(kept)

    os.makedirs('audit', exist_ok=True)
    with open('audit/removed_heldout_edges.json', 'w', encoding='utf-8') as f:
        json.dump(removed, f, indent=2, ensure_ascii=False)

    manifest = {
        'dataset': str(ds),
        'original_edges': len(removed) + len(kept),
        'kept_edges': len(kept),
        'removed_edges': len(removed),
        'path_graph_sha256': sha256_file(path_graph),
        'path_graph_train_only_sha256': sha256_file(out_path),
        'removed_heldout_edges': 'audit/removed_heldout_edges.json',
    }
    with open('audit/sanitized_graph_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'original={manifest["original_edges"]} kept={manifest["kept_edges"]} '
          f'removed={manifest["removed_edges"]}')
    print(f'written: {out_path} / audit/removed_heldout_edges.json / audit/sanitized_graph_manifest.json')


if __name__ == '__main__':
    main()
