# Case-Study Evidence (Dataset 2, FINAL): PubMed 独立文献佐证（人工复核材料）

- 生成日期: 2026-08-23
- 来源预测 CSV: `predictions_ds2_retrained_0shot.csv`（tier=test2, y_true=1, 5 训练种子: 19940419 / 20230801 / 20240520 / 20260201 / 20260301，Dataset 2 上重训）
- 选择规则: 每 (drug_a, drug_b, event) 在 5 个训练种子上取 prob/uncertainty 均值；r = p_mean·(1−u_mean)；剔除与 Dataset 1 药对级重叠的 1068 个 test2 药对（dataset2_pair_overlap.json）；按 r 降序取 **pre-registered top-20**（R24），**全部 20 个候选均列于此表，无剔除、无挑选**。
- 药名解析: DB ID → drug_smiles → InChIKey-14 → RxPairEvid 名字表（ddi_pairs_50k.csv）；未解析的药对用 `DrugBank {DB_ID}` 作 PubMed 查询词（dataset2 无名字文件，controller ruling b）。
- **semantic_overlap 标志（R26）**: “yes” = 该候选事件与 Dataset 1 某事件 max cosine ≥ 0.7（即不在 disjoint_events.json）；“no” = 事件在 10 个语义不相交事件中 （max cosine < 0.7）。本表 2 个候选为 “no”。
- **启发式佐证计数: 0/20**（标题同时提及两药查询词且含交互/不良反应语境关键词；仅作人工复核提示，**非最终科学判断**；FAERS 信号统计不视为独立证据，且 dataset2 无 FAERS 数据，相关列为空）
- 检索式: `"a_name"[All Fields] AND "b_name"[All Fields] AND (interaction OR adverse)`（未解析名字时追加事件文本关键词），PubMed Entrez，top-3 PMID+标题，0.4s/请求限速。
- **triage/referral 框架（R24）**: 未获文献佐证的候选不剔除，仍列于表中，标注为需人工优先复核的 triage/referral 候选。最终案例结论由作者裁决。

## Rank 1 — Dexniguldipine + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB09239` / `DB00388`（DrugBank id）; a_name=Dexniguldipine, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6852, u_mean=0.6295, r=0.2538
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 2 — Temazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB00231` / `DB00388`（DrugBank id）; a_name=Temazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6793, u_mean=0.6415, r=0.2435
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.

## Rank 3 — Phenylephrine + Trimebutine（The risk or severity of hypertension decrease）
- 药对: `DB00388` / `DB09089`（DrugBank id）; a_name=Phenylephrine, b_name=Trimebutine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6787, u_mean=0.6426, r=0.2426
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 4 — Oxazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB00842` / `DB00388`（DrugBank id）; a_name=Oxazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6766, u_mean=0.6467, r=0.2390
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.

## Rank 5 — Clotiazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB01559` / `DB00388`（DrugBank id）; a_name=Clotiazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6731, u_mean=0.6539, r=0.2330
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 6 — Nitrazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB01595` / `DB00388`（DrugBank id）; a_name=Nitrazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6723, u_mean=0.6555, r=0.2316
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.

## Rank 7 — Dotarizine + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB06446` / `DB00388`（DrugBank id）; a_name=Dotarizine, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6707, u_mean=0.6586, r=0.2290
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 8 — Ketazolam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB01587` / `DB00388`（DrugBank id）; a_name=Ketazolam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6706, u_mean=0.6588, r=0.2288
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 9 — Diazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB00829` / `DB00388`（DrugBank id）; a_name=Diazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6700, u_mean=0.6601, r=0.2277
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.
  2. [21492386](https://pubmed.ncbi.nlm.nih.gov/21492386/) — Arrhythmias and transient changes in cardiac function after topical administration of one drop of phenylephrine 10% in an adult cat undergoing conjunctival graft.
  3. [11675045](https://pubmed.ncbi.nlm.nih.gov/11675045/) — Synergistic interaction of diazepam with 3',5'-cyclic adenosine monophosphate-elevating agents on rat aortic rings.

## Rank 10 — Lorazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB00186` / `DB00388`（DrugBank id）; a_name=Lorazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6691, u_mean=0.6619, r=0.2262
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.

## Rank 11 — Phenylephrine + Tetrahydropalmatine（The risk or severity of hypertension decrease）
- 药对: `DB00388` / `DB12093`（DrugBank id）; a_name=Phenylephrine, b_name=Tetrahydropalmatine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6684, u_mean=0.6631, r=0.2252
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 12 — Flurazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB00690` / `DB00388`（DrugBank id）; a_name=Flurazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6680, u_mean=0.6640, r=0.2245
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.
  2. [11468017](https://pubmed.ncbi.nlm.nih.gov/11468017/) — Biotransformation of xenobiotics by amine oxidases.
  3. [3370486](https://pubmed.ncbi.nlm.nih.gov/3370486/) — Electrophysiological actions of norepinephrine in rat lateral hypothalamus. II. An in vitro study of the effects of iontophoretically applied norepinephrine on LH neuronal responses to gamma-aminobutyric acid (GABA).

## Rank 13 — Levodopa + Trimebutine（The absorption decrease）
- 药对: `DB01235` / `DB09089`（DrugBank id）; a_name=Levodopa, b_name=Trimebutine
- 事件: The absorption decrease
- 模型输出（5 种子均值）: prob_mean=0.6673, u_mean=0.6655, r=0.2232
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 14 — Flunitrazepam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB01544` / `DB00388`（DrugBank id）; a_name=Flunitrazepam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6670, u_mean=0.6661, r=0.2227
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.
  2. [1595911](https://pubmed.ncbi.nlm.nih.gov/1595911/) — Hemodynamic effects of anesthesia in patients chronically treated with angiotensin-converting enzyme inhibitors.
  3. [2854841](https://pubmed.ncbi.nlm.nih.gov/2854841/) — Differential responsiveness of cerebellar Purkinje neurons to GABA and benzodiazepine receptor ligands in an animal model of hepatic encephalopathy.

## Rank 15 — Agmatine + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB08838` / `DB00388`（DrugBank id）; a_name=Agmatine, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6663, u_mean=0.6673, r=0.2217
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [28837867](https://pubmed.ncbi.nlm.nih.gov/28837867/) — The inhibition of inducible nitric oxide synthase and oxidative stress by agmatine attenuates vascular dysfunction in rat acute endotoxemic model.

## Rank 16 — Etizolam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB09166` / `DB00388`（DrugBank id）; a_name=Etizolam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6654, u_mean=0.6691, r=0.2202
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.

## Rank 17 — Levodopa + Propiverine（The absorption decrease）
- 药对: `DB01235` / `DB12278`（DrugBank id）; a_name=Levodopa, b_name=Propiverine
- 事件: The absorption decrease
- 模型输出（5 种子均值）: prob_mean=0.6650, u_mean=0.6699, r=0.2195
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 18 — Clobazam + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB00349` / `DB00388`（DrugBank id）; a_name=Clobazam, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6646, u_mean=0.6708, r=0.2188
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [33220270](https://pubmed.ncbi.nlm.nih.gov/33220270/) — Vasorelaxant effects of benzodiazepines, non-benzodiazepine sedative-hypnotics, and tandospirone on isolated rat arteries.

## Rank 19 — Phenylephrine + Manidipine（The risk or severity of hypertension decrease）
- 药对: `DB00388` / `DB09238`（DrugBank id）; a_name=Phenylephrine, b_name=Manidipine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6632, u_mean=0.6736, r=0.2165
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  - （无检索结果）

## Rank 20 — Desipramine + Desipramine（The serum concentration of the active metabolites increase）
- 药对: `DB01151` / `DB01151`（DrugBank id）; a_name=Desipramine, b_name=Desipramine
- 事件: The serum concentration of the active metabolites increase
- 模型输出（5 种子均值）: prob_mean=0.6629, u_mean=0.6742, r=0.2160
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- 佐证标志: NO（未获独立文献佐证；triage/referral 候选）
- PubMed 检索结果:
  1. [28520379](https://pubmed.ncbi.nlm.nih.gov/28520379/) — Imipramine Therapy and CYP2D6 and CYP2C19 Genotype.
  2. [42381757](https://pubmed.ncbi.nlm.nih.gov/42381757/) — The efficacy and safety of hyoscyamine, dicyclomine, and desipramine in the treatment of irritable pouch syndrome-a retrospective cohort study.
  3. [42015492](https://pubmed.ncbi.nlm.nih.gov/42015492/) — Methylphenidate and Psychotic Symptoms in Children and Adolescents: A Disproportionality Analysis on the WHO Safety Database (VigiBase).
