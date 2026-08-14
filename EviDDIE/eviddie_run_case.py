#!/usr/bin/env python
"""Retrospective Examples from Held-Out Test Split: inspects high-scoring positive instances
from Dataset 2 test2 split against DrugBank records. This is a descriptive consistency check,
NOT independent external validation or novel interaction discovery."""
import json, logging, numpy as np, torch, torch.nn.functional as F
import os
from collections import defaultdict
from tqdm import tqdm
from eviddie_args import read_options
from eviddie_dataloader import DrugDataset, DrugDataLoader
from eviddie_matcher import EmbedMatcher, Generate_Model
from shared.checkpoint import load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = 'models/dataset2/bestmodels'
G_PATH = 'models/dataset2/bestmodels_G'
DATASET = 'dataset2'

# ---- Init (same as export scripts) ----
arg = read_options()
arg.dataset = DATASET
arg.semantic = 'event_embedding2.json'

sm = json.load(open(f'{DATASET}/{arg.semantic}'))
# P0-7: inference uses the raw BioSentVec embeddings (no semantic noise).
for t in sm: sm[t] = np.array(sm[t])
te_list, t2id = [], {}
for num,i in enumerate(list(sm.keys())):
    t2id[i]=num; te_list.append(sm[i])
te = torch.tensor(np.vstack(te_list)).float().to(device)

# Load symbols
sid={}; r2=json.load(open(f'{DATASET}/relation2ids')); e2=json.load(open(f'{DATASET}/ent2ids'))
r2e=json.load(open(f'{DATASET}/relation2embids')); e2e=json.load(open(f'{DATASET}/ent2embids'))
ee=np.load(f'{DATASET}/DRKG_TransE_entity.npy'); re=np.load(f'{DATASET}/DRKG_TransE_relation.npy')
i=0; emb=[]
for k in r2:
    if k not in ['','OOV']: sid[k]=i; i+=1; emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
for k in e2:
    if k not in ['','OOV']: sid[k]=i; i+=1; emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
sid['PAD']=i; emb.append(list(np.zeros((re.shape[1],))))
sv = np.array(emb)

matcher = EmbedMatcher(128, len(sid)-1, use_pretrain=True, embed=sv, dropout=0.2, batch_size=256, finetune=True, aggregate='max', task_emb=te).to(device).eval()

ckpt = torch.load(CKPT, map_location=device)
# P0-7: the case-study conclusions must come from native dual-output EDL
# checkpoints; legacy 1-output checkpoints are refused instead of converted.
if 'fc.5.weight' in ckpt and ckpt['fc.5.weight'].shape[0] != 2:
    raise RuntimeError(
        'Legacy 1-output checkpoint detected (fc.5.weight shape != 2). '
        'Retrain with eviddie_trainer.py (native dual-output EDL head; see README).')
for k in list(ckpt.keys()):
    if any(x in k for x in ['symbol_emb','gcn_w','gcn_b','Bilinear','Linear_self','Linear_nei','Linear_weak_rel','NeighborAggregator','siamese','support_encoder','query_encoder']):
        del ckpt[k]
load_state_dict_safe(matcher, ckpt, model_name='matcher')

G_m = Generate_Model(in_dim=te.shape[1]).to(device)
G_m = torch.load(G_PATH, map_location=device)
G_m.eval()

e1re2 = defaultdict(list); e1re2.update(json.load(open(f'{DATASET}/e1rel_e2.json')))
rel2c = json.load(open(f'{DATASET}/rel2candidates.json'))
rel2id = r2
ent2id = e2

# ---- 审计：加载 train/dev/test split 的有序和无序药物对 ----
# 注意：不加载 test2，因为 test2 正例是本次回顾性分析的目标候选本身
all_pairs = set()
for split_name in ['train_tasks', 'dev_tasks', 'test_tasks']:
    path = f'{DATASET}/{split_name}.json'
    if not os.path.exists(path): continue
    tasks = json.load(open(path))
    for evt, triples in tasks.items():
        for t in triples:
            all_pairs.add((t[0], t[2]))
            all_pairs.add((t[2], t[0]))
logging.info(f'Loaded {len(all_pairs)} unique drug pairs from all splits')

# ---- DrugBank pairs ----
db_pairs = set()
dti_path = f'{DATASET}/dti_entity.csv'
if os.path.exists(dti_path):
    with open(dti_path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 2:
                db_pairs.add((p[0].strip(), p[1].strip()))
logging.info(f'Loaded {len(db_pairs)} DrugBank pairs')

# ---- Run inference on test2 (rare-event) tasks ----
test2 = json.load(open(f'{DATASET}/test2_tasks.json'))
results = []  # (prob, unc, score_r, event_type, d1, d2, in_db, in_split)

with torch.no_grad():
    for evt, triples in tqdm(test2.items(), desc='Inference'):
        if not triples: continue
        cand = rel2c[evt]
        np.random.seed(2024)
        ft = []
        for t in triples:
            eh,rel,et = t[0],t[1],t[2]
            while True:
                n = np.random.choice(cand)
                if n not in e1re2.get(eh+rel,[]) and n!=et: break
            ft.append([eh,rel,n])
        at = triples + ft
        ar = [[t[0],t[2],rel2id[t[1]]] for t in at]
        npos = len(triples)
        qb = DrugDataset(ar)
        qbl = DrugDataLoader(qb, batch_size=len(ar), shuffle=False)
        qbd = [t.to(device) for t in next(iter(qbl))]
        proto = G_m(te[t2id[evt]]).detach()
        ql,qr = matcher.model(qbd)
        qn = torch.cat((ql,qr),-1)
        _,_,_,zq = matcher.vaemodel(qn, is_support=False, is_eval=True)
        fc_out = matcher.fc(torch.abs(proto.expand_as(zq)-zq))
        ev = F.softplus(fc_out); al = ev+1
        prob = al[:,1]/al.sum(1); unc = 2.0/al.sum(1)

        for idx in range(npos):
            d1,d2 = triples[idx][0], triples[idx][2]
            p = prob[idx].item(); u = unc[idx].item()
            r = p * (1-u)
            in_db = (d1,d2) in db_pairs or (d2,d1) in db_pairs
            in_split = (d1,d2) in all_pairs or (d2,d1) in all_pairs
            if not in_split:  # only candidates NOT in any data split
                results.append((p, u, r, evt, d1, d2, in_db))

# Sort by prob descending
results.sort(key=lambda x: x[0], reverse=True)
logging.info(f'Total candidates (cross-split filtered): {len(results)}')

# ---- Output top-K ----
print(f'\n{"="*100}')
print(f'Retrospective Examples from Held-Out Test Split (Table 5)')
print(f'Total retained test2 positive instances after excluding train/dev/test overlaps: {len(results)}')
print(f'{"="*100}')
print(f'{"Rank":<6} {"Drug1":<10} {"Drug2":<10} {"Event":<40} {"Prob":<10} {"Uncertainty":<12} {"Score r":<10} {"Source":<20} {"Action":<15}')
print('-'*100)

τp, τu = 0.75, 0.40  # default thresholds
K = min(10, len(results))
for rank in range(K):
    p,u,r,evt,d1,d2,in_db = results[rank]
    source = 'DrugBank' if in_db else 'Not identified'
    action = 'High-priority' if (p>=τp and u<τu) else 'Expert referral' if (p>=τp and u>=τu) else 'Deferred'
    print(f'{rank+1:<6} {d1:<10} {d2:<10} {evt[:40]:<40} {p:<10.4f} {u:<12.4f} {r:<10.4f} {source:<20} {action:<15}')

# ---- LaTeX output ----
print(f'\n\nLATEX TABLE:')
print(r'\begin{table}[htbp]')
print(r'\centering')
print(r'\caption{Retrospective examples from the held-out test2 split of Dataset~2. Instances are held-out positive records from the benchmark; this analysis is descriptive and does not constitute independent external validation.}')
print(r'\label{tab:case_study}')
print(r'\scriptsize')
print(r'\setlength{\tabcolsep}{2pt}')
print(r'\begin{tabularx}{\textwidth}{cllp{2.5cm}cccc}')
print(r'\toprule')
print(r'\textbf{Rank} & \textbf{Drug 1} & \textbf{Drug 2} & \textbf{Event type} & \textbf{Prob.} & \textbf{Uncertainty} & \textbf{Score $r$} & \textbf{Source} & \textbf{Action} \\')
print(r'\midrule')
for rank in range(K):
    p,u,r,evt,d1,d2,in_db = results[rank]
    source = 'DrugBank' if in_db else 'Not identified'
    action = 'High-priority' if (p>=τp and u<τu) else 'Expert referral' if (p>=τp and u>=τu) else 'Deferred'
    evt_short = evt.replace('#Drug1', 'Drug1').replace('#Drug2', 'Drug2')[:50]
    print(f'{rank+1} & {d1} & {d2} & {evt_short} & {p:.4f} & {u:.4f} & {r:.4f} & {source} & {action} \\\\')
print(r'\bottomrule')
print(r'\end{tabularx}')
print(r'\end{table}')
