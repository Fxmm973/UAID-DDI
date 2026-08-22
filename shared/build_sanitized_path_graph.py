#!/usr/bin/env python
# coding=utf-8
import argparse
import hashlib
import json
import os
from pathlib import Path

MANIFEST_PATH = os.path.join('audit', 'sanitized_graph_manifest.json')


def normalized_pair(a, b):
    return tuple(sorted((a, b)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm_key(path):
    return os.path.normpath(str(path)).replace(os.sep, '/')


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding='utf-8') as f:
            manifest = json.load(f)
        if 'graphs' not in manifest:  # legacy single-dataset manifest
            ds = manifest.get('dataset', '').replace('\\', '/')
            graphs = {}
            if manifest.get('path_graph_sha256'):
                graphs[f'{ds}/path_graph'] = manifest['path_graph_sha256']
            if manifest.get('path_graph_train_only_sha256'):
                graphs[f'{ds}/path_graph_train_only'] = manifest['path_graph_train_only_sha256']
            meta_keys = ('original_edges', 'kept_edges', 'removed_edges', 'removed_heldout_edges')
            datasets = {ds: {k: v for k, v in manifest.items() if k in meta_keys}}
            return {'graphs': graphs, 'datasets': datasets}
        return manifest
    return {'graphs': {}, 'datasets': {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='PharDDIE/dataset1')
    args = ap.parse_args()
    ds = Path(args.dataset)
    repo = Path(__file__).resolve().parent.parent
    ds_key = norm_key(ds.resolve().relative_to(repo))

    heldout_pairs = set()
    for split in ['test', 'test2']:
        tasks_path = ds / f'{split}_tasks.json'
        if not tasks_path.exists():
            print(f'[WARN] {tasks_path} missing; skipping its held-out pairs')
            continue
        tasks = json.load(open(tasks_path, encoding='utf-8'))
        for event, rows in tasks.items():
            for head, relation, tail in rows:
                heldout_pairs.add(normalized_pair(head, tail))
    print(f'held-out drug pairs: {len(heldout_pairs)}')

    path_graph = ds / 'path_graph'
    if not path_graph.exists():
        raise FileNotFoundError(f'{path_graph} not found; run the graph build step first.')

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
    removed_file = 'audit/removed_heldout_edges_' + ds_key.replace('/', '_') + '.json'
    with open(removed_file, 'w', encoding='utf-8') as f:
        json.dump(removed, f, indent=2, ensure_ascii=False)

    manifest = load_manifest()
    manifest['graphs'][f'{ds_key}/path_graph'] = sha256_file(path_graph)
    manifest['graphs'][f'{ds_key}/path_graph_train_only'] = sha256_file(out_path)
    manifest['datasets'][ds_key] = {
        'original_edges': len(removed) + len(kept),
        'kept_edges': len(kept),
        'removed_edges': len(removed),
        'removed_heldout_edges': removed_file,
    }
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'original={manifest["datasets"][ds_key]["original_edges"]} '
          f'kept={manifest["datasets"][ds_key]["kept_edges"]} '
          f'removed={manifest["datasets"][ds_key]["removed_edges"]}')
    print(f'written: {out_path} / {removed_file} / {MANIFEST_PATH}')


if __name__ == '__main__':
    main()
