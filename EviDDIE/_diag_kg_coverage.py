import json
from collections import Counter

DS = 'dataset1'
ent2id = json.load(open(DS + '/ent2ids'))

kg_heads = set()
with open(DS + '/path_graph') as f:
    for line in f:
        kg_heads.add(line.rstrip('\n').split('\t')[0][-7:])

def coverage(path):
    data = json.load(open(DS + '/' + path))
    triples = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                triples.extend(v)
    else:
        triples = data
    drugs = set()
    for t in triples:
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            drugs.add(t[0]); drugs.add(t[1])
    n = len(drugs)
    with_kg = sum(1 for d in drugs if d in kg_heads)
    return n, with_kg, with_kg / max(n, 1)

for p in ['train_tasks.json', 'dev_tasks.json', 'test_tasks.json', 'test2_tasks.json']:
    n, w, r = coverage(p)
    print(f'{p}: drugs={n} with-KG={w} ratio={r:.3f}')
