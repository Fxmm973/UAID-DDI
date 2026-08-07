"""Debug: check why case study returns 0 results"""
import json, os

DATASET = 'dataset2'

# 1. Load test2 tasks
test2 = json.load(open(f'{DATASET}/test2_tasks.json'))
print(f'test2 events: {len(test2)}')
total_triples = sum(len(v) for v in test2.values())
print(f'total triples in test2: {total_triples}')

# 2. Check first event
first_evt = list(test2.keys())[0]
first_triples = test2[first_evt]
print(f'\nFirst event: {first_evt}')
print(f'First 3 triples: {first_triples[:3]}')

# 3. Check all_pairs from train+dev+test
all_pairs = set()
for split_name in ['train_tasks', 'dev_tasks', 'test_tasks']:
    path = f'{DATASET}/{split_name}.json'
    if os.path.exists(path):
        tasks = json.load(open(path))
        for evt, triples in tasks.items():
            for t in triples:
                all_pairs.add((t[0], t[2]))
                all_pairs.add((t[2], t[0]))
print(f'\nUnique pairs in train+dev+test: {len(all_pairs)}')

# 4. Check how many test2 pairs are in all_pairs
test2_pairs = set()
for evt, triples in test2.items():
    for t in triples:
        test2_pairs.add((t[0], t[2]))
overlap = test2_pairs & all_pairs
print(f'test2 unique pairs: {len(test2_pairs)}')
print(f'test2 pairs also in train+dev+test: {len(overlap)}')
print(f'test2 pairs NOT in train+dev+test: {len(test2_pairs - overlap)}')

# 5. Check DrugDataset import
from eviddie_dataloader import DrugDataset, DrugDataLoader
from shared.preprocess import MOL_EDGE_LIST_FEAT_MTX, DRUG_INDX_NAME_DICT
print(f'\nMOL_EDGE_LIST_FEAT_MTX keys: {len(MOL_EDGE_LIST_FEAT_MTX)}')
print(f'DRUG_INDX_NAME_DICT keys: {len(DRUG_INDX_NAME_DICT)}')

# 6. Check if test2 drug IDs are in MOL
d1, d2, _ = first_triples[0]
print(f'\nFirst triple drugs: {d1}, {d2}')
print(f'  {d1} in MOL: {d1 in MOL_EDGE_LIST_FEAT_MTX}')
print(f'  {d2} in MOL: {d2 in MOL_EDGE_LIST_FEAT_MTX}')

# 7. Try creating a DrugDataset
rel2id = json.load(open(f'{DATASET}/relation2ids'))
ar = [[first_triples[0][0], first_triples[0][2], rel2id[first_triples[0][1]]]]
print(f'\nDrugDataset input: {ar}')
qb = DrugDataset(ar)
print(f'DrugDataset len: {len(qb)}')
