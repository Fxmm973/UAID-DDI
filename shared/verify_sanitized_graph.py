#!/usr/bin/env python
# coding=utf-8
import hashlib
import json
import os

_REPO_AUDIT_CANDIDATES = None


def _find_manifest():
    global _REPO_AUDIT_CANDIDATES
    if _REPO_AUDIT_CANDIDATES is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _REPO_AUDIT_CANDIDATES = [os.path.join(here, '..', 'audit', 'sanitized_graph_manifest.json')]
    for p in _REPO_AUDIT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        'audit/sanitized_graph_manifest.json not found; run shared/build_sanitized_path_graph.py first.')


def _norm_key(path):
    return os.path.normpath(path).replace(os.sep, '/')


def _repo_key(dataset_dir):
    """Repository-relative key for a dataset dir, independent of the calling cwd."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_abs = os.path.abspath(os.path.join(os.getcwd(), dataset_dir))
    rel = os.path.relpath(ds_abs, repo)
    if rel.startswith('..'):
        raise RuntimeError(
            f'dataset dir {dataset_dir} resolves outside the repository ({ds_abs}); '
            f'cannot verify its sanitized graph.')
    return _norm_key(rel)


def verify_sanitized_graph(dataset_dir, graph_file='path_graph'):
    """Fail-closed check (P0-7): the neighbor-index graph must be the sanitized
    file recorded in audit/sanitized_graph_manifest.json. Missing record or
    hash mismatch raises and aborts the run."""
    manifest = json.load(open(_find_manifest(), encoding='utf-8'))
    graphs = manifest.get('graphs')
    if graphs is None:  # legacy single-dataset manifest
        ds = manifest.get('dataset', '').replace('\\', '/')
        graphs = {f'{ds}/path_graph': manifest.get('path_graph_sha256'),
                  f'{ds}/path_graph_train_only': manifest.get('path_graph_train_only_sha256')}
    key = f'{_repo_key(dataset_dir)}/{graph_file}'
    recorded = graphs.get(key)
    if recorded is None:
        raise RuntimeError(
            f'No sanitized-graph hash recorded for {key} in sanitized_graph_manifest.json. '
            f'Run shared/build_sanitized_path_graph.py --dataset {dataset_dir} first.')
    graph_path = os.path.join(dataset_dir, graph_file)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f'Sanitized graph not found: {graph_path}.')
    actual = hashlib.sha256(open(graph_path, 'rb').read()).hexdigest()
    if actual != recorded:
        raise RuntimeError(
            f'Sanitized graph hash mismatch for {graph_path}: '
            f'recorded={recorded} actual={actual}. '
            f'Refusing to proceed (fail closed). Re-run shared/build_sanitized_path_graph.py.')
    return True
