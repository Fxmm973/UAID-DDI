#!/usr/bin/env python
# coding=utf-8
"""为 Table 5 (Case Study) 生成补充字段：Event, Uncertainty, Score r, Triage action"""
import json, numpy as np, torch, torch.nn.functional as F

# === 用户需确认的配置 ===
DATASET = 'dataset2'
CKPT_DIR = 'models/dataset1'  # EviDDIE checkpoint 目录
TAU_P, TAU_U = 0.75, 0.40  # 阈值（用验证集选的，这里用默认值，需确认）
DRUGBANK_VERSION = 'DrugBank v5.1.12 (accessed 2025-03)'

# === Table 5 现有的 10 个候选对（从 TeX 提取）===
candidates = [
    (1, 'DB00795', 'DB01708', "Decreased Prasterone excretion (higher serum level)."),
    (2, 'DB01250', 'DB04574', "Increased liver damage risk: Olsalazine + Estrone sulfate."),
    (3, 'DB00864', 'DB01008', "Increased immunosuppressive activity: Tacrolimus + Busulfan."),
    (4, 'DB08991', 'DB09095', "Increased GI irritation: Difluocortolone + Epirizole."),
    (5, 'DB08439', 'DB00959', "Increased GI irritation: Methylprednisolone + Parecoxib."),
    (6, 'DB00091', 'DB00624', "Increased liver damage risk: Testosterone + Cyclosporine."),
    (7, 'DB01628', 'DB04865', "Increased bleeding risk: Etoricoxib + Omacetaxine."),
    (8, 'DB03585', 'DB00288', "No external evidence identified."),
    (9, 'DB04743', 'DB01410', "Increased GI irritation: Ciclesonide + Nimesulide."),
    (10, 'DB00881', 'DB01592', "Decreased adverse effects: Iron + Quinapril."),
]

# === 跨 split 审计 ===
def load_all_pairs(dataset_dir):
    """加载所有 split 的药物对集合"""
    all_pairs = set()
    for split in ['train_tasks', 'dev_tasks', 'test_tasks', 'test2_tasks']:
        path = f'{dataset_dir}/{split}.json'
        if not os.path.exists(path):
            print(f'  Skip {split} (not found)')
            continue
        tasks = json.load(open(path))
        for event, triples in tasks.items():
            for t in triples:
                all_pairs.add((t[0], t[2]))  # (d_i, d_j)
                all_pairs.add((t[2], t[0]))  # (d_j, d_i) reverse
    return all_pairs

import os
dataset_dir = f'{DATASET}'  # EviDDIE/dataset2
pairs_in_data = load_all_pairs(dataset_dir)
print(f'Loaded {len(pairs_in_data)} unique drug pairs from all splits')

# === DrugBank 查找 ===
drugbank_pairs = set()
dti_path = f'{dataset_dir}/dti_entity.csv'
if os.path.exists(dti_path):
    with open(dti_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                drugbank_pairs.add((parts[0].strip(), parts[1].strip()))

print(f'Loaded {len(drugbank_pairs)} pairs from DrugBank (dti_entity.csv)')

# === 检查每个候选 ===
print(f'\n{"="*100}')
print('Table 5: Post Hoc Database Concordance Examples (Audit Results)')
print(f'{"="*100}')
print(f'{"Rank":<6} {"Drug1":<10} {"Drug2":<10} {"In any split?":<16} {"In DrugBank?":<16} {"Evidence source":<30}')

for rank, d1, d2, desc in candidates:
    in_split = (d1, d2) in pairs_in_data or (d2, d1) in pairs_in_data
    in_db = (d1, d2) in drugbank_pairs or (d2, d1) in drugbank_pairs

    if in_split:
        evidence = 'EXCLUDED (appears in data split)'
    elif in_db:
        evidence = DRUGBANK_VERSION
    else:
        evidence = 'Not identified'

    print(f'{rank:<6} {d1:<10} {d2:<10} {"YES" if in_split else "no":<16} {"YES" if in_db else "no":<16} {evidence:<30}')

# === 输出 Table 5 的 LaTeX（含新列）===
print(f'\n\n{"="*100}')
print('LATEX TABLE (with new columns: Event, Uncertainty, Score r, Source, Action)')
print(f'{"="*100}')

print(r"""\begin{table}[htbp]
\centering
\caption{Post hoc database concordance examples: top-10 predictions from the rare-event split of Dataset~2.}
\label{tab:case_study}
\scriptsize
\renewcommand{\arraystretch}{1.1}
\setlength{\tabcolsep}{2pt}
\begin{tabularx}{\textwidth}{cllp{2.5cm}cccc}
\toprule
\textbf{Rank} & \textbf{Drug 1} & \textbf{Drug 2} & \textbf{Event type} & \textbf{Prob.} & \textbf{Uncertainty} & \textbf{Score $r$} & \textbf{Source} & \textbf{Action} \\
\midrule""")

for rank, d1, d2, desc in candidates:
    in_split = (d1, d2) in pairs_in_data or (d2, d1) in pairs_in_data
    in_db = (d1, d2) in drugbank_pairs or (d2, d1) in drugbank_pairs

    # Placeholders for model outputs (需要实际推理才能填)
    prob = "[P]"
    unc = "[U]"
    score = "[R]"
    event_type = "[Event]"

    if in_split:
        source = 'EXCLUDED'
        action = '—'
    elif in_db:
        source = 'DrugBank'
        # 根据 triage 规则
        # action = 'High-priority' if p>=τp and u<τu else 'Expert referral'
        action = '[Action]  # based on p, u, τp, τu'
    else:
        source = 'Not identified'
        action = 'Expert referral'

    print(f'{rank} & {d1} & {d2} & {event_type} & {prob} & {unc} & {score} & {source} & {action} \\\\')

print(r"""\bottomrule
\end{tabularx}
\end{table}""")
