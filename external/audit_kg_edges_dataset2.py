#!/usr/bin/env python
# coding=utf-8
"""Task 12 (Step 2): KG-edge overlap audit for dataset2 test2 — REPORT ONLY.

Counts how many of the 1872 test2 (d_i, d_j) unordered pairs appear as an
edge endpoint pair in EviDDIE/dataset2/path_graph_train_only and/or
path_graph (the KG context the zero-shot export reads). The graph edge ids
are prefixed ("Compound::DB00001", "Gene::2147", ...); the comparison uses the
prefix-stripped id so a drug pair (DB00001, DB00002) can match an edge whose
endpoints are exactly {DB00001, DB00002}. The graph is never modified.

Output: external/outputs/dataset2_kg_edge_overlap.json
"""
import hashlib
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATASET2 = os.path.join(REPO, 'EviDDIE', 'dataset2')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


def load_test_pairs():
    tasks = json.load(open(os.path.join(DATASET2, 'test2_tasks.json'), encoding='utf-8'))
    pairs = set()
    n_triples = 0
    for ev, triples in tasks.items():
        n_triples += len(triples)
        for t in triples:
            pairs.add(tuple(sorted((t[0], t[2]))))
    return pairs, n_triples


def load_edge_pairs(path):
    """Unordered, prefix-stripped endpoint pairs of all graph edges."""
    pairs = set()
    n_edges = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            e1, _rel, e2 = line.rstrip().split('\t')
            pairs.add(tuple(sorted((e1.split('::')[-1], e2.split('::')[-1]))))
            n_edges += 1
    return pairs, n_edges


def sha256_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    test_pairs, n_triples = load_test_pairs()
    tr_pairs, n_tr = load_edge_pairs(os.path.join(DATASET2, 'path_graph_train_only'))
    pg_pairs, n_pg = load_edge_pairs(os.path.join(DATASET2, 'path_graph'))
    ovl_tr = sorted(test_pairs & tr_pairs)
    ovl_pg = sorted(test_pairs & pg_pairs)
    tr_lines = set(open(os.path.join(DATASET2, 'path_graph_train_only'), encoding='utf-8').read().splitlines())
    pg_lines = set(open(os.path.join(DATASET2, 'path_graph'), encoding='utf-8').read().splitlines())
    same_edge_set = tr_lines == pg_lines

    summary = {
        'n_test2_triples': n_triples,
        'n_test2_unordered_pairs': len(test_pairs),
        'path_graph_train_only': {
            'n_edges': n_tr,
            'n_distinct_unordered_endpoint_pairs': len(tr_pairs),
            'n_drug_drug_edges': sum(1 for p in tr_pairs
                                     if all(x.startswith('DB') for x in p)),
            'sha256': sha256_file(os.path.join(DATASET2, 'path_graph_train_only')),
        },
        'path_graph': {
            'n_edges': n_pg,
            'n_distinct_unordered_endpoint_pairs': len(pg_pairs),
            'n_drug_drug_edges': sum(1 for p in pg_pairs
                                     if all(x.startswith('DB') for x in p)),
            'sha256': sha256_file(os.path.join(DATASET2, 'path_graph')),
        },
        'overlap_train_only': len(ovl_tr),
        'overlap_train_only_rate': round(len(ovl_tr) / len(test_pairs), 4),
        'overlap_train_only_pairs': ovl_tr,
        'overlap_path_graph': len(ovl_pg),
        'overlap_path_graph_rate': round(len(ovl_pg) / len(test_pairs), 4),
        'overlap_path_graph_pairs': ovl_pg,
        'path_graph_files_same_edge_set': same_edge_set,
        'report_only': True,
        'note': ('REPORT ONLY — the graph was not modified. DRKG contains no '
                 'drug-drug edges, so no test2 (drug, drug) pair can be an edge; '
                 'the overlap reflects whether any edge endpoint pair equals a '
                 'tested drug pair.'),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'dataset2_kg_edge_overlap.json')
    json.dump(summary, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
