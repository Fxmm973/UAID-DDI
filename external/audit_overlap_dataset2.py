#!/usr/bin/env python
# coding=utf-8
"""Task 12 (Step 4): dataset2 vs Dataset 1 overlap audit (drug-level and pair-level).

a. Drug-level: the 1258 dataset2 drugs (SMILES -> InChIKey-14, RDKit) vs the
   1706 dataset1 drugs -> n, rate, list (dataset2_drug_overlap.json).
b. Pair-level: the 1872 test2 unordered drug pairs (DB ids, as in the task
   files) vs the pairs appearing in dataset1's train/dev/test/test2 task
   files -> n, rate, list, plus a per-split breakdown
   (dataset2_pair_overlap.json).

Also reports the encodability pre-check numbers that the zero-shot export
needs (all dataset2 drugs present in dataset2 ent2ids, SMILES parseable,
molecule-feature registrable) so that dropped_pairs=0 is justified.

Reuses audit_overlap_ext.ik14_of (the reviewed Task-9 function).
"""
import json
import os

import pandas as pd
from rdkit import Chem

from audit_overlap_ext import ik14_of

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
DS1 = os.path.join(REPO, 'PharDDIE', 'dataset1')
DS2 = os.path.join(REPO, 'EviDDIE', 'dataset2')
DS1_TASK_FILES = ['train_tasks', 'dev_tasks', 'test_tasks', 'test2_tasks']


def load_drug_smiles(csv_path):
    df = pd.read_csv(csv_path, dtype={'drug_id': 'string'})
    return df.dropna(subset=['drug_id'])


def drug_level_overlap():
    ds2 = load_drug_smiles(os.path.join(DS2, 'drug_smiles.csv'))
    ds1 = load_drug_smiles(os.path.join(DS1, 'drug_smiles.csv'))
    ds2['ik14'] = ds2['smiles'].map(ik14_of)
    ds1['ik14'] = ds1['smiles'].map(ik14_of)
    ds2_valid = ds2.dropna(subset=['ik14'])
    ds1_valid = ds1.dropna(subset=['ik14'])
    ds2_set = set(ds2_valid['ik14'])
    ds1_set = set(ds1_valid['ik14'])
    overlap_ik14 = sorted(ds2_set & ds1_set)
    # DB ids of dataset2 drugs whose IK14 is shared with a dataset1 drug
    overlap_db = sorted(ds2_valid.loc[ds2_valid['ik14'].isin(ds1_set), 'drug_id'])
    return {
        'n_dataset2_drugs': int(len(ds2)),
        'n_dataset2_drugs_valid_ik14': int(len(ds2_valid)),
        'n_dataset2_drugs_invalid_smiles': int(len(ds2) - len(ds2_valid)),
        'n_dataset1_drugs': int(len(ds1)),
        'n_dataset1_drugs_valid_ik14': int(len(ds1_valid)),
        'n_overlap_ik14': len(overlap_ik14),
        'overlap_rate_of_dataset2': round(len(overlap_ik14) / max(1, len(ds2_valid)), 4),
        'overlap_rate_of_dataset1': round(len(overlap_ik14) / max(1, len(ds1_valid)), 4),
        'overlap_ik14': overlap_ik14,
        'overlap_dataset2_db_ids': overlap_db,
        'invalid_smiles_db_ids': sorted(ds2.loc[ds2['ik14'].isna(), 'drug_id']),
    }


def dataset1_task_pairs():
    pairs = set()
    per_split = {}
    for f in DS1_TASK_FILES:
        tasks = json.load(open(os.path.join(DS1, f + '.json'), encoding='utf-8'))
        sp = set()
        for ev, triples in tasks.items():
            for t in triples:
                p = tuple(sorted((t[0], t[2])))
                sp.add(p)
                pairs.add(p)
        per_split[f] = {'n_events': len(tasks), 'n_pairs': len(sp)}
    return pairs, per_split


def pair_level_overlap():
    tasks = json.load(open(os.path.join(DS2, 'test2_tasks.json'), encoding='utf-8'))
    ds2_pairs = set()
    n_triples = 0
    per_event = {}
    for ev, triples in tasks.items():
        n_triples += len(triples)
        ev_pairs = set()
        for t in triples:
            p = tuple(sorted((t[0], t[2])))
            ds2_pairs.add(p)
            ev_pairs.add(p)
        per_event[ev] = {'n_triples': len(triples), 'n_distinct_pairs': len(ev_pairs)}
    ds1_pairs, ds1_per_split = dataset1_task_pairs()
    overlap = sorted(ds2_pairs & ds1_pairs)
    return {
        'n_test2_triples': n_triples,
        'n_test2_distinct_unordered_pairs': len(ds2_pairs),
        'n_dataset1_pairs_over_all_splits': len(ds1_pairs),
        'dataset1_pair_counts_per_split': ds1_per_split,
        'n_overlap_pairs': len(overlap),
        'overlap_rate': round(len(overlap) / max(1, len(ds2_pairs)), 4),
        'overlap_pairs': [list(p) for p in overlap],
        'test2_event_triple_counts': per_event,
    }


def encodability_precheck():
    """All dataset2 drugs must be ent2ids keys with parseable SMILES for the
    zero-shot export to have zero drops (dropped_pairs=0)."""
    ent2ids = json.load(open(os.path.join(DS2, 'ent2ids'), encoding='utf-8'))
    ds2 = load_drug_smiles(os.path.join(DS2, 'drug_smiles.csv'))
    drugs = set(ds2['drug_id'])
    in_ent2ids = sum(1 for d in drugs if d in ent2ids)
    parseable = sum(1 for smi in ds2['smiles'] if Chem.MolFromSmiles(str(smi)) is not None)
    return {
        'n_dataset2_drugs': len(drugs),
        'in_ent2ids': in_ent2ids,
        'missing_from_ent2ids': sorted(drugs - set(ent2ids)),
        'smiles_parseable': parseable,
        'smiles_unparseable': len(drugs) - parseable,
    }


def main():
    drug = drug_level_overlap()
    pair = pair_level_overlap()
    enc = encodability_precheck()
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(drug, open(os.path.join(OUT_DIR, 'dataset2_drug_overlap.json'), 'w'),
              indent=2, ensure_ascii=False)
    json.dump(pair, open(os.path.join(OUT_DIR, 'dataset2_pair_overlap.json'), 'w'),
              indent=2, ensure_ascii=False)
    json.dump(enc, open(os.path.join(OUT_DIR, 'dataset2_encodability_precheck.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('DRUG OVERLAP:', json.dumps({k: v for k, v in drug.items()
                                       if not k.endswith('ids') and 'ik14' not in k or k.startswith('n_')},
                                      indent=2))
    print('PAIR OVERLAP:', json.dumps({k: v for k, v in pair.items()
                                       if k != 'overlap_pairs' and k != 'test2_event_triple_counts'},
                                      indent=2))
    print('ENCODABILITY:', json.dumps(enc, indent=2))
    print(f'wrote {OUT_DIR}/dataset2_drug_overlap.json, dataset2_pair_overlap.json, '
          f'dataset2_encodability_precheck.json')


if __name__ == '__main__':
    main()
