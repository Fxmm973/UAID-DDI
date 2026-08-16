#!/usr/bin/env python
# coding=utf-8
"""Retrospective Examples from Held-Out Test Split (Table 5).
Loads EviDDIE checkpoint, runs inference on Dataset 2 test2 positives,
performs cross-split audit and DrugBank lookup, outputs LaTeX table.

This analysis is DESCRIPTIVE — it inspects held-out benchmark positives,
not unknown drug pairs. DrugBank is used only for post hoc annotation.
This is NOT independent external validation or novel interaction discovery.
"""
import json, logging, numpy as np, torch, torch.nn.functional as F
import os
from collections import defaultdict
from eviddie_args import read_options
from eviddie_dataloader import DrugDataset, DrugDataLoader
from eviddie_matcher import EmbedMatcher, Generate_Model
from shared.checkpoint import convert_fc_1to2, load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 用户需确认的配置 ===
DATASET = 'dataset2'
CKPT = 'models/dataset2/bestmodels'
G_PATH = 'models/dataset2/bestmodels_G'
TAU_P, TAU_U = 0.75, 0.40  # 阈值（从验证集网格搜索选定，需与实际实验一致）
DRUGBANK_VERSION = 'DrugBank 5.0 (Wishart et al., Nucleic Acids Res., 2018)'

# ---- Init matcher + generator (复用 export 脚本逻辑) ----
arg = read_options()
arg.dataset = DATASET
arg.semantic = 'event_embedding2.json'

sm = json.load(open(f'{DATASET}/{arg.semantic}'))
for t in sm:
    sm[t] = np.array(sm[t]) + 0.3 * np.random.normal(0, 1, size=(len(sm[t]), 1))
te_list, t2id = [], {}
for num, i in enumerate(list(sm.keys())):
    t2id[i] = num
    te_list.append(sm[i])
te = torch.tensor(np.vstack(te_list)).float().to(device)

sid = {}
r2 = json.load(open(f'{DATASET}/relation2ids'))
e2 = json.load(open(f'{DATASET}/ent2ids'))
r2e = json.load(open(f'{DATASET}/relation2embids'))
e2e = json.load(open(f'{DATASET}/ent2embids'))
ee = np.load(f'{DATASET}/DRKG_TransE_entity.npy')
re = np.load(f'{DATASET}/DRKG_TransE_relation.npy')
i = 0
emb = []
for k in r2:
    if k not in ['', 'OOV']:
        sid[k] = i; i += 1
        emb.append(list(re[r2e[k], :]) if r2e[k] != -1 else list(np.random.randn(re.shape[1])))
for k in e2:
    if k not in ['', 'OOV']:
        sid[k] = i; i += 1
        emb.append(list(ee[e2e[k], :]) if e2e[k] != -1 else list(np.random.randn(re.shape[1])))
sid['PAD'] = i
emb.append(list(np.zeros((re.shape[1],))))
sv = np.array(emb)

matcher = EmbedMatcher(128, len(sid) - 1, use_pretrain=True, embed=sv, dropout=0.2,
                        batch_size=256, finetune=True, aggregate='max',
                        task_emb=te).to(device).eval()

ckpt = torch.load(CKPT, map_location=device)
convert_fc_1to2(ckpt)
for k in list(ckpt.keys()):
    if any(x in k for x in ['symbol_emb', 'gcn_w', 'gcn_b', 'Bilinear', 'Linear_self',
                             'Linear_nei', 'Linear_weak_rel', 'NeighborAggregator',
                             'siamese', 'support_encoder', 'query_encoder']):
        del ckpt[k]
load_state_dict_safe(matcher, ckpt, model_name='matcher')

G_m = Generate_Model(in_dim=te.shape[1]).to(device)
G_m = torch.load(G_PATH, map_location=device)
G_m.eval()

e1re2 = defaultdict(list)
e1re2.update(json.load(open(f'{DATASET}/e1rel_e2.json')))
rel2c = json.load(open(f'{DATASET}/rel2candidates.json'))
rel2id = r2

# ---- 跨 split 审计：加载 train/dev/test 中的有序和无序药物对 ----
# test2 正例本身不加入排除集，因为它们是本次回顾性分析的目标候选
all_pairs = set()
for split_name in ['train_tasks', 'dev_tasks', 'test_tasks']:
    path = f'{DATASET}/{split_name}.json'
    if not os.path.exists(path):
        logging.warning(f'Split not found: {path}')
        continue
    tasks = json.load(open(path))
    for evt, triples in tasks.items():
        for t in triples:
            all_pairs.add((t[0], t[2]))
            all_pairs.add((t[2], t[0]))
logging.info(f'Loaded {len(all_pairs)} unique drug pairs from train/dev/test splits')

# ---- DrugBank 一致性查找：加载全量 DDI 记录用于事后注释 ----
# DrugBank 是基准标签的来源，因此从任务 JSON 和 e1rel_e2 中加载所有已知正例
db_pairwise = set()       # (drug_a, drug_b) — 药物对级别匹配
db_event_specific = set() # (drug_a, event, drug_b) — 事件级别匹配
for split_name in ['train_tasks', 'dev_tasks', 'test_tasks']:
    path = f'{DATASET}/{split_name}.json'
    if not os.path.exists(path):
        continue
    tasks = json.load(open(path))
    for evt, triples in tasks.items():
        for t in triples:
            db_pairwise.add((t[0], t[2]))
            db_pairwise.add((t[2], t[0]))
            db_event_specific.add((t[0], evt, t[2]))
for key, val_list in e1re2.items():
    for val in val_list:
        for evt in rel2c:
            if key.endswith(evt):
                d1 = key[:-len(evt)]
                db_pairwise.add((d1, val))
                db_pairwise.add((val, d1))
                db_event_specific.add((d1, evt, val))
                break

logging.info(f'Loaded {len(db_pairwise)} unique (unordered) drug pairs from all DDI records')
logging.info(f'Loaded {len(db_event_specific)} event-specific (drug, event, drug) triples')

def check_drugbank_evidence(d1, d2, evt):
    """返回 DrugBank 证据级别：
    'DrugBank' — 药物对和事件类型均与 DrugBank 记录一致
    'DrugBank (pair only)' — 药物对存在于 DrugBank，但事件类型不匹配
    'Not identified' — 未在 DrugBank 中找到该药物对
    """
    pair_match = (d1, d2) in db_pairwise or (d2, d1) in db_pairwise
    event_match = (d1, evt, d2) in db_event_specific
    if event_match:
        return 'DrugBank'
    elif pair_match:
        return 'DrugBank (pair only)'
    else:
        return 'Not identified'

# ---- 在 test2 正例上推理 ----
test2 = json.load(open(f'{DATASET}/test2_tasks.json'))
results = []

with torch.no_grad():
    for evt, triples in test2.items():
        if not triples:
            continue
        cand = rel2c[evt]
        np.random.seed(2024)
        ft = []
        for t in triples:
            eh, rel, et = t[0], t[1], t[2]
            while True:
                n = np.random.choice(cand)
                if n not in e1re2.get(eh + rel, []) and n != et:
                    break
            ft.append([eh, rel, n])
        at = triples + ft
        ar = [[t[0], t[2], rel2id[t[1]]] for t in at]
        npos = len(triples)
        qb = DrugDataset(ar)
        qbl = DrugDataLoader(qb, batch_size=len(ar), shuffle=False)
        qbd = [t.to(device) for t in next(iter(qbl))]
        proto = G_m(te[t2id[evt]]).detach()
        ql, qr = matcher.model(qbd)
        qn = torch.cat((ql, qr), -1)
        _, _, _, zq = matcher.vaemodel(qn, is_support=False, is_eval=True)
        fc_out = matcher.fc(torch.abs(proto.expand_as(zq) - zq))
        ev = F.softplus(fc_out)
        al = ev + 1
        prob = al[:, 1] / al.sum(1)
        unc = 2.0 / al.sum(1)

        for idx in range(npos):
            d1, d2 = triples[idx][0], triples[idx][2]
            p = prob[idx].item()
            u = unc[idx].item()
            r = p * (1 - u)
            evidence = check_drugbank_evidence(d1, d2, evt)
            in_split = (d1, d2) in all_pairs or (d2, d1) in all_pairs
            if not in_split:
                results.append((p, u, r, evt, d1, d2, evidence))

results.sort(key=lambda x: x[0], reverse=True)
logging.info(f'Total retained test2 positive instances: {len(results)}')

# ---- 输出 ----
print(f'\n{"=" * 100}')
print('Retrospective Examples from Held-Out Test Split (Table 5)')
print(f'Total retained instances (after excluding train/dev/test overlaps): {len(results)}')
print(f'DrugBank version: {DRUGBANK_VERSION}')
print(f'Triage thresholds: tau_p={TAU_P}, tau_u={TAU_U}')
print(f'{"=" * 100}')
print(f'{"Rank":<6} {"Drug1":<10} {"Drug2":<10} {"Event":<45} {"Prob":<10} {"Uncertainty":<12} {"Score r":<10} {"DB evidence":<24} {"Action":<15}')
print('-' * 110)

K = min(8, len(results))
for rank in range(K):
    p, u, r, evt, d1, d2, evidence = results[rank]
    if p >= TAU_P and u < TAU_U:
        action = 'High-priority review'
    elif p >= TAU_P and u >= TAU_U:
        action = 'Expert referral'
    elif p < TAU_P and u >= TAU_U:
        action = 'Deferred review'
    else:
        action = 'Low priority'
    print(f'{rank + 1:<6} {d1:<10} {d2:<10} {evt[:45]:<45} {p:<10.4f} {u:<12.4f} {r:<10.4f} {evidence:<24} {action:<15}')

# ---- LaTeX 输出 ----
print(f'\n\nLATEX TABLE:')
print(r'\begin{table}[htbp]')
print(r'\centering')
print(r'\caption{Retrospective examples from the held-out test2 split of Dataset~2. '
      r'Instances are held-out positive records from the benchmark. '
      r'DrugBank (' + DRUGBANK_VERSION + r') was used only for post hoc annotation. '
      r'This analysis is descriptive and does not constitute independent external validation.}')
print(r'\label{tab:case_study}')
print(r'\scriptsize')
print(r'\renewcommand{\arraystretch}{1.1}')
print(r'\setlength{\tabcolsep}{2pt}')
print(r'\begin{tabularx}{\textwidth}{cllp{2.5cm}cccc}')
print(r'\toprule')
print(r'\textbf{Rank} & \textbf{Drug 1} & \textbf{Drug 2} & \textbf{Event type} & '
      r'\textbf{Prob.} & \textbf{Uncertainty} & \textbf{Score $r$} & \textbf{DB match} & \textbf{Action} \\')
print(r'\midrule')
for rank in range(K):
    p, u, r, evt, d1, d2, evidence = results[rank]
    if p >= TAU_P and u < TAU_U:
        action = 'High-priority'
    elif p >= TAU_P and u >= TAU_U:
        action = 'Expert referral'
    else:
        action = 'Deferred'
    evt_short = evt[:60]
    print(f'{rank + 1} & {d1} & {d2} & {evt_short} & {p:.4f} & {u:.4f} & {r:.4f} & {evidence} & {action} \\\\')
print(r'\bottomrule')
print(r'\end{tabularx}')
print(r'\end{table}')
