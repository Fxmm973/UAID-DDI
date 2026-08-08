#!/usr/bin/env python
# coding=utf-8
"""
Generate fixed negative-sample manifests for each split and seed.
Output: JSON files with pre-computed negative drug IDs for every positive query triple.
Also computes SHA256 hashes and runs event-level overlap audit.
"""
import json
import hashlib
import random
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]


def load_indexes(dataset):
    """Load necessary data files."""
    e1rel_e2 = json.load(open(f'{dataset}/e1rel_e2.json'))
    rel2candidates = json.load(open(f'{dataset}/rel2candidates.json'))
    return e1rel_e2, rel2candidates


def generate_manifest(split_tasks, e1rel_e2, rel2candidates, seed, output_path):
    """
    Generate fixed negative samples for every positive query triple.
    For each (d_i, e, d_j) positive:
      - Sample d_k from C_e s.t. d_k != d_j and (d_i, d_k, e) not in known positives.
    Store as {event: [[d_i, d_j, d_k, rel], ...]}.
    """
    random.seed(seed)
    manifest = {}

    for event, triples in tqdm(split_tasks.items(), desc=f'Seed {seed}'):
        candidates = rel2candidates[event]
        event_negatives = []
        for triple in triples:
            d_i, rel, d_j = triple[0], triple[1], triple[2]
            while True:
                d_k = random.choice(candidates)
                if d_k != d_j and (d_k not in e1rel_e2.get(d_i + rel, [])):
                    break
            event_negatives.append([d_i, d_j, d_k, rel])
        manifest[event] = event_negatives

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # SHA256 hash
    with open(output_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash


def audit_overlap(manifest_path, split_tasks, split_name):
    """
    Audit: check that no support-query overlap and no positive-negative overlap exist.
    """
    manifest = json.load(open(manifest_path))
    issues = []

    for event, neg_list in manifest.items():
        pos_pairs = set()
        for triple in split_tasks.get(event, []):
            pos_pairs.add((triple[0], triple[2]))  # (d_i, d_j)

        neg_pairs = set()
        for entry in neg_list:
            d_i, d_j, d_k, rel = entry
            neg_pairs.add((d_i, d_k))

        # Check positive-negative overlap
        overlap = pos_pairs & neg_pairs
        if overlap:
            issues.append(f'[{split_name}] Event {event}: {len(overlap)} positive-negative overlaps')

        # Check within-manifest dedup
        if len(neg_pairs) != len(neg_list):
            issues.append(f'[{split_name}] Event {event}: duplicate negatives in manifest')

    if issues:
        print(f'  AUDIT ISSUES FOUND ({len(issues)}):')
        for issue in issues:
            print(f'    - {issue}')
    else:
        print(f'  AUDIT PASSED: no overlaps or duplicates.')

    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset1', type=str)
    args = parser.parse_args()
    dataset = args.dataset

    e1rel_e2, rel2candidates = load_indexes(dataset)
    splits = {
        'dev': json.load(open(f'{dataset}/dev_tasks.json')),
        'test': json.load(open(f'{dataset}/test_tasks.json')),
        'test2': json.load(open(f'{dataset}/test2_tasks.json')),
    }

    manifest_dir = f'{dataset}/neg_manifests'
    import os
    os.makedirs(manifest_dir, exist_ok=True)

    hash_log = {}
    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'Generating manifests for seed={seed}')
        print(f'{"="*60}')
        for split_name, tasks in splits.items():
            output_path = f'{manifest_dir}/{split_name}_seed{seed}_negatives.json'
            file_hash = generate_manifest(tasks, e1rel_e2, rel2candidates, seed, output_path)
            hash_log[f'{split_name}_seed{seed}'] = {'path': output_path, 'sha256': file_hash}
            print(f'  {split_name}: {output_path}')
            print(f'  SHA256: {file_hash}')

            # Run overlap audit
            audit_overlap(output_path, tasks, split_name)

    # Save hash log
    hash_log_path = f'{manifest_dir}/manifest_hashes.json'
    with open(hash_log_path, 'w') as f:
        json.dump(hash_log, f, indent=2)
    print(f'\nHash log saved to {hash_log_path}')

    # Print summary for TeX
    print(f'\n{"="*60}')
    print('SUMMARY FOR TEX:')
    print(f'Total events: dev={len(splits["dev"])}, test={len(splits["test"])}, test2={len(splits["test2"])}')
    print(f'Negative manifests generated for {len(SEEDS)} seeds across {len(splits)} splits.')
    print(f'Manifest directory: {manifest_dir}/')
    print(f'Hash log: {hash_log_path}')


if __name__ == '__main__':
    main()
