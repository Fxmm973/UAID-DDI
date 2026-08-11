#!/usr/bin/env python
# coding=utf-8
"""
Drug Overlap Audit: quantifies how many test-set drugs appear in the training set.
This determines whether the evaluation setting is:
  - "unseen-event generalization" (drugs overlap but event types are new)
  - "novel-compound generalization" (test drugs genuinely unseen during training)
"""
import json
import argparse


def load_drugs_from_tasks(tasks_path):
    drugs = set()
    tasks = json.load(open(tasks_path))
    for evt, triples in tasks.items():
        for t in triples:
            drugs.add(t[0])
            drugs.add(t[2])
    return drugs


def audit_overlap(dataset_path):
    splits = {}
    for split_name in ['train_tasks', 'dev_tasks', 'test_tasks', 'test2_tasks']:
        path = f'{dataset_path}/{split_name}.json'
        try:
            splits[split_name] = load_drugs_from_tasks(path)
        except FileNotFoundError:
            print(f'  WARNING: {path} not found, skipping.')

    print('=' * 70)
    print(f'Drug Overlap Audit: {dataset_path}')
    print('=' * 70)
    for split_name, drugs in splits.items():
        print(f'  {split_name}: {len(drugs)} unique drugs')

    train_drugs = splits.get('train_tasks', set())
    dev_drugs = splits.get('dev_tasks', set())
    test_drugs = splits.get('test_tasks', set())
    test2_drugs = splits.get('test2_tasks', set())

    print()
    print('--- Overlap with Training Set ---')
    for label, split_drugs in [('dev', dev_drugs), ('test (fewer)', test_drugs),
                                ('test2 (rare)', test2_drugs)]:
        if not split_drugs:
            continue
        overlap = split_drugs & train_drugs
        novel = split_drugs - train_drugs
        print(f'  {label}: {len(split_drugs)} drugs total')
        print(f'    In training:   {len(overlap)} ({100*len(overlap)/len(split_drugs):.1f}%)')
        print(f'    Novel (not in training): {len(novel)} ({100*len(novel)/len(split_drugs):.1f}%)')
        if novel:
            print(f'    Novel drug IDs: {sorted(list(novel))[:20]}')

    print()
    all_eval = set()
    if test_drugs:
        all_eval |= test_drugs
    if test2_drugs:
        all_eval |= test2_drugs
    if all_eval:
        novel_eval = all_eval - train_drugs
        novel_pct = 100 * len(novel_eval) / len(all_eval)
        print(f'  Combined evaluation: {len(all_eval)} drugs, {len(novel_eval)} novel ({novel_pct:.1f}%)')
        print()
        print('--- Generalization Claim Guidance ---')
        if novel_pct > 30:
            print('  Can claim "novel-compound generalization"')
        elif novel_pct > 5:
            print('  Mixed: some novel drugs. Recommend "unseen-event generalization')
            print('  with partial novel-compound coverage"')
        else:
            print('  Must claim "unseen-event generalization" only.')
            print('  The setting evaluates generalization to unseen INTERACTION TYPES,')
            print('  not generalization to unseen DRUGS.')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PharDDIE/dataset1')
    args = parser.parse_args()
    audit_overlap(args.dataset)
