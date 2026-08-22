# Case-Study Evidence (Dataset 2): PubMed 独立文献佐证（人工复核材料）

- 生成日期: 2026-08-22
- 来源预测 CSV: `predictions_dataset2_eviddie_0shot.csv`（tier=test2, y_true=1, 5 种子）
- 选择规则: 每 (drug_a, drug_b, event) 在 5 个训练种子上取 prob/uncertainty 均值；r = p_mean·(1−u_mean)；剔除与 Dataset 1 药对级重叠的 1068 个 test2 药对（dataset2_pair_overlap.json）；按 r 降序取 top-10。
- 药名解析: DB ID → drug_smiles → InChIKey-14 → RxPairEvid 名字表（ddi_pairs_50k.csv）；未解析的药对用 `DrugBank {DB_ID}` 作 PubMed 查询词（dataset2 无名字文件，controller ruling b）。
- **启发式佐证计数: 0/10**（标题同时提及两药查询词且含交互/不良反应语境关键词；仅作人工复核提示，非最终科学判断；FAERS 信号统计不视为独立证据）
- 检索式: `"a_name"[All Fields] AND "b_name"[All Fields] AND (interaction OR adverse)`（未解析名字时追加事件文本关键词），PubMed Entrez，top-3 PMID+标题，0.4s/请求限速。

## 证据列说明（论文表格用）
- 独立证据列：PubMed PMID（唯一独立佐证来源，R12）
- 非独立证据列：FAERS 统计（标签源自 FAERS，仅作背景摘录，不计入佐证计数）

## Rank 1 — Aprobarbital + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB01352` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5764, u_mean=0.7758, r=0.1293
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 2 — Barbital + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB01483` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5789, u_mean=0.7787, r=0.1281
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 3 — Thiopental + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB00599` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5761, u_mean=0.7787, r=0.1275
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 4 — Methylphenobarbital + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB00849` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5762, u_mean=0.7790, r=0.1273
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 5 — Secobarbital + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB00418` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5770, u_mean=0.7801, r=0.1269
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 6 — Flurazepam + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB00690` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5740, u_mean=0.7815, r=0.1254
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 7 — Hexobarbital + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB01355` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5763, u_mean=0.7824, r=0.1254
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 8 — Nitrazepam + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB01595` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5732, u_mean=0.7822, r=0.1248
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 9 — Temazepam + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB00231` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5743, u_mean=0.7828, r=0.1248
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 10 — Triazolam + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB00897` / `DB12161`（DrugBank id）
- 模型输出（5 种子均值）: prob_mean=0.5750, u_mean=0.7834, r=0.1245
- FAERS（非独立证据，仅摘录）: n_reports=None, PRR_max_strict=None, ROR95_lcl_max_strict=None
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）
