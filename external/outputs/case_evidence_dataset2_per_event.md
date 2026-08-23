# Case-Study Evidence (Dataset 2, 每事件最优候选 + 三档证据): 人工复核材料

- 生成日期: 2026-08-23
- 来源预测 CSV: `predictions_ds2_retrained_0shot.csv`（tier=test2, y_true=1, 5 训练种子: 19940419 / 20230801 / 20240520 / 20260201 / 20260301，Dataset 2 上重训）
- 选择规则（R28）: 每个 test2 事件取 **1 个最优候选** = 该事件内 y_true=1、r 最高（r = p_mean·(1−u_mean)，5 种子均值）的 (drug_a, drug_b, event) 三元组；剔除与 Dataset 1 药对级重叠的 1068 个 test2 药对（dataset2_pair_overlap.json）。**25/25 个事件均有候选 → 25 行，全部列出，无挑选、无剔除。**
- 药名解析: DB ID → drug_smiles → InChIKey-14 → RxPairEvid 名字表（ddi_pairs_50k.csv）；未解析的药对用 `DrugBank {DB_ID}` 作 PubMed 查询词（dataset2 无名字文件，controller ruling b）。
- **semantic_overlap 标志（R26）**: “yes” = 该候选事件与 Dataset 1 某事件 max cosine ≥ 0.7（不在 disjoint_events.json）；“no” = 事件在 10 个语义不相交事件中（cos < 0.7）。本表 10/25 个候选为 “no”。
- **三档证据统计（R29，自动判定，人工复核材料）: direct 2 / class_suggested 2 / none 21（共 25 候选）**
- 检索: `"a_name"[All Fields] AND "b_name"[All Fields]`（retmax 5；esearch + esummary 标题 + efetch 摘要；0.4s/请求限速，失败自动重试）。
- **Direct 档**（保守正则）: 任一摘要中两药名（或其查询词）出现在同一句/±150 字符窗口内，且窗口含交互/不良反应语境词 ('interact', 'advers', 'drug-drug', 'ddi', 'combination', 'concomitant', 'toxicity', 'overdose', 'side effect', 'additive', 'synergis', 'potentiat')。
- **Class-level 建议档**: 摘要含至少一个药名，且命中该事件机制词表（按事件文本选词：血管舒张 vasorelax*/vasodilat*/hypotens*/blood pressure；吸收 absorb*/bioavailab；代谢 metaboli*；浓度 serum/plasma concentration；CNS 抑制 sedat*/CNS depress* 等）；输出命中句片段供人工裁决。
- 其余 = **Not identified**。**自动结果仅为人工复核材料；最终 Evidence 列由作者裁决（R29）。FAERS 不构成独立证据（R12），dataset2 亦无 FAERS 数据。**
- 注: `rank` = 该候选在其事件内的排名（每事件恰取 1 个最优 → 恒为 1）；行序按 r 全局降序。

## Dexniguldipine + Phenylephrine（The risk or severity of hypertension decrease）
- 药对: `DB09239` / `DB00388`（DrugBank id）; a_name=Dexniguldipine, b_name=Phenylephrine
- 事件: The risk or severity of hypertension decrease
- 模型输出（5 种子均值）: prob_mean=0.6852, u_mean=0.6295, r=0.2538
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Levodopa + Trimebutine（The absorption decrease）
- 药对: `DB01235` / `DB09089`（DrugBank id）; a_name=Levodopa, b_name=Trimebutine
- 事件: The absorption decrease
- 模型输出（5 种子均值）: prob_mean=0.6673, u_mean=0.6655, r=0.2232
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — 检索到 2 篇，其中 2 篇有摘要；均未通过 Direct 判定，也未见药名+机制词共现 → Not identified
- PubMed 检索结果（retmax=5）:
  - [27956826](https://pubmed.ncbi.nlm.nih.gov/27956826/) — Nonmotor gastrointestinal disorders in older patients with Parkinson's disease: is there hope?
  - [15909927](https://pubmed.ncbi.nlm.nih.gov/15909927/) — Examination of antimicrobial activity of selected non-antibiotic drugs.
  - （未检出 direct / class-level 证据）

## Desipramine + Desipramine（The serum concentration of the active metabolites increase）
- 药对: `DB01151` / `DB01151`（DrugBank id）; a_name=Desipramine, b_name=Desipramine
- 事件: The serum concentration of the active metabolites increase
- 模型输出（5 种子均值）: prob_mean=0.6629, u_mean=0.6742, r=0.2160
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: class_suggested（类别级机制建议，需人工裁决）** — 摘要 42549822 含至少一个药名且命中事件机制词 ['active metabolite', 'plasma concentration', 'metaboli']；命中句片段见下（供人工裁决）
- PubMed 检索结果（retmax=5）:
  - [42587754](https://pubmed.ncbi.nlm.nih.gov/42587754/) — Stepwise Translational Validation of the Screening Hit Desipramine Reveals Limits of Fibroblast-State Modulation in Lung Fibrosis.
  - [42549822](https://pubmed.ncbi.nlm.nih.gov/42549822/) — A pharmacometric framework for norepinephrine transporter occupancy and dose equivalence across psychotropic medications.
  - [42538543](https://pubmed.ncbi.nlm.nih.gov/42538543/) — Beyond rodents: A systematic review of antidepressant-like effects in Drosophila melanogaster as an alternative model in psychopharmacology.
  - [28520379](https://pubmed.ncbi.nlm.nih.gov/28520379/) — Imipramine Therapy and CYP2D6 and CYP2C19 Genotype.
  - [42470701](https://pubmed.ncbi.nlm.nih.gov/42470701/) — From Visceral Pain to Emotional Distress: Comparative Effectiveness of Antispasmodics and Antidepressants in Irritable Bowel Syndrome - Systematic Review and Network Meta-analysis.
  - Class-level 命中句 1/2（PMID 42549822，机制词 ['active metabolite', 'plasma concentration', 'metaboli']）: “To facilitate cross-drug comparisons of noradrenergic activity, we developed a pharmacometric model to estimate NET occupancy for 26 psychotropic agents and their active metabolites.”
  - Class-level 命中句 2/2（PMID 42549822，机制词 ['active metabolite', 'plasma concentration', 'metaboli']）: “NET occupancy was estimated using National Institute of Mental Health Psychoactive Drug Screening Program Ki data, protein-binding-corrected plasma concentrations, a standard receptor occupancy model, and logit-derived ED50 values, and was compared with published positron emission tomography (PET) estimates.”

## Terbutaline + Pipecuronium（the neuromuscular blocking activities decrease）
- 药对: `DB00871` / `DB01338`（DrugBank id）; a_name=Terbutaline, b_name=Pipecuronium
- 事件: the neuromuscular blocking activities decrease
- 模型输出（5 种子均值）: prob_mean=0.6517, u_mean=0.6966, r=0.1977
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Ertugliflozin + Quinapril（The risk or severity of renal failure hypotension and hyperkalemia increase）
- 药对: `DB11827` / `DB00881`（DrugBank id）; a_name=Ertugliflozin, b_name=Quinapril
- 事件: The risk or severity of renal failure hypotension and hyperkalemia increase
- 模型输出（5 种子均值）: prob_mean=0.6513, u_mean=0.6974, r=0.1971
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## DrugBank DB14006 + Ramipril（The risk or severity of renal failure increase）
- 药对: `DB14006` / `DB00178`（DrugBank id）; a_name=(未解析), b_name=Ramipril
- 事件: The risk or severity of renal failure increase
- 模型输出（5 种子均值）: prob_mean=0.6450, u_mean=0.7101, r=0.1870
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Erythromycin + Dexniguldipine（The risk or severity of QTc prolongation and hypotension increase）
- 药对: `DB00199` / `DB09239`（DrugBank id）; a_name=Erythromycin, b_name=Dexniguldipine
- 事件: The risk or severity of QTc prolongation and hypotension increase
- 模型输出（5 种子均值）: prob_mean=0.6233, u_mean=0.7533, r=0.1538
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Lumateperone + Lisdexamfetamine（the stimulatory activities decrease）
- 药对: `DB06077` / `DB01255`（DrugBank id）; a_name=Lumateperone, b_name=Lisdexamfetamine
- 事件: the stimulatory activities decrease
- 模型输出（5 种子均值）: prob_mean=0.6216, u_mean=0.7568, r=0.1512
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — 检索到 1 篇，其中 1 篇有摘要；均未通过 Direct 判定，也未见药名+机制词共现 → Not identified
- PubMed 检索结果（retmax=5）:
  - [39096466](https://pubmed.ncbi.nlm.nih.gov/39096466/) — Pharmacological Treatment of Binge Eating Disorder and Frequent Comorbid Diseases.
  - （未检出 direct / class-level 证据）

## Sitagliptin + Alogliptin（The risk or severity of angioedema increase）
- 药对: `DB01261` / `DB06203`（DrugBank id）; a_name=Sitagliptin, b_name=Alogliptin
- 事件: The risk or severity of angioedema increase
- 模型输出（5 种子均值）: prob_mean=0.6163, u_mean=0.7674, r=0.1434
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: direct（直接证据）** — 摘要 42312164 中两药查询词出现在同一句/±150 字符窗口内且含交互/不良反应语境词（('interact', 'advers', 'drug-drug', 'ddi', 'combination', 'concomitant', 'toxicity', 'overdose', 'side effect', 'additive', 'synergis', 'potentiat')）；片段: “In silico analyses, including molecular docking as well as molecular dynamics simulation, demonstrate good binding affinity as well as stable interaction of vildagliptin with PI3K (4YKN) and NLRP3 (7ALV) proteins in comparison to other DPP-4 inhibitors (sitagliptin, saxagliptin, linagliptin, and alogliptin).”
- PubMed 检索结果（retmax=5）:
  - [42614024](https://pubmed.ncbi.nlm.nih.gov/42614024/) — Dipeptidyl Peptidase-4 Inhibitors Associated Heart Failure Events in Adult Patients With Type-2 Diabetes Mellitus Treated With Dipeptidyl Peptidase-4 Inhibitors: A Systematic Review and Meta-Analysis.
  - [42443801](https://pubmed.ncbi.nlm.nih.gov/42443801/) — Comparison of the permeability of DPP-4 inhibitors-sitagliptin, vildagliptin, linagliptin, and alogliptin-in placental barrier cell models and exploration of their transport mechanisms by LC-MS/MS.
  - [42348590](https://pubmed.ncbi.nlm.nih.gov/42348590/) — Early potential safety signals for gliptins and gliflozins using real-world pharmacy data compared to spontaneous reporting.
  - [42312164](https://pubmed.ncbi.nlm.nih.gov/42312164/) — Phosphatidylinositol-3-kinase/Protein Kinase B (PI3K/AKT) and Nucleotide-Binding Oligomerization Domain-like Receptor Family Pyrin Domain Containing 3 (NLRP3) Inflammasome Modulation Underlies the Neuroprotective Effects of Vildagliptin in a Rotenone-Induced Mouse Model of Parkinson's Disease.
  - [32809447](https://pubmed.ncbi.nlm.nih.gov/32809447/) — EMS Diabetic Protocols for Treat and Release.
  - Direct 证据片段（PMID 42312164）: “In silico analyses, including molecular docking as well as molecular dynamics simulation, demonstrate good binding affinity as well as stable interaction of vildagliptin with PI3K (4YKN) and NLRP3 (7ALV) proteins in comparison to other DPP-4 inhibitors (sitagliptin, saxagliptin, linagliptin, and alogliptin).”

## Pyrantel + Budesonide（The risk or severity of myopathy and weakness increase）
- 药对: `DB11156` / `DB01222`（DrugBank id）; a_name=Pyrantel, b_name=Budesonide
- 事件: The risk or severity of myopathy and weakness increase
- 模型输出（5 种子均值）: prob_mean=0.3411, u_mean=0.5817, r=0.1427
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## (+)-Mefloquine + Befunolol（The risk or severity of QTc prolongation decrease）
- 药对: `DB00358` / `DB09013`（DrugBank id）; a_name=(+)-Mefloquine, b_name=Befunolol
- 事件: The risk or severity of QTc prolongation decrease
- 模型输出（5 种子均值）: prob_mean=0.6062, u_mean=0.7876, r=0.1288
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Niflumic acid + Mestranol（the thrombogenic activities increase）
- 药对: `DB04552` / `DB01357`（DrugBank id）; a_name=Niflumic acid, b_name=Mestranol
- 事件: the thrombogenic activities increase
- 模型输出（5 种子均值）: prob_mean=0.2528, u_mean=0.5054, r=0.1250
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Bendroflumethiazide + Belinostat（The risk or severity of neutropenia and thrombocytopenia increase）
- 药对: `DB00436` / `DB05015`（DrugBank id）; a_name=Bendroflumethiazide, b_name=Belinostat
- 事件: The risk or severity of neutropenia and thrombocytopenia increase
- 模型输出（5 种子均值）: prob_mean=0.2599, u_mean=0.5197, r=0.1248
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Cidoxepin + Deutetrabenazine（The risk or severity of sedation and somnolence increase）
- 药对: `DB01142` / `DB12161`（DrugBank id）; a_name=Cidoxepin, b_name=Deutetrabenazine
- 事件: The risk or severity of sedation and somnolence increase
- 模型输出（5 种子均值）: prob_mean=0.2850, u_mean=0.5699, r=0.1226
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Trimebutine + Nabilone（The risk or severity of Tachycardia and drowsiness increase）
- 药对: `DB09089` / `DB00486`（DrugBank id）; a_name=Trimebutine, b_name=Nabilone
- 事件: The risk or severity of Tachycardia and drowsiness increase
- 模型输出（5 种子均值）: prob_mean=0.3106, u_mean=0.6211, r=0.1177
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Ergometrine + Tianeptine（the vasopressor activities increase）
- 药对: `DB01253` / `DB09289`（DrugBank id）; a_name=Ergometrine, b_name=Tianeptine
- 事件: the vasopressor activities increase
- 模型输出（5 种子均值）: prob_mean=0.5952, u_mean=0.8096, r=0.1133
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Amitriptyline + Irinotecan（The risk or severity of neutropenia increase）
- 药对: `DB00321` / `DB00762`（DrugBank id）; a_name=Amitriptyline, b_name=Irinotecan
- 事件: The risk or severity of neutropenia increase
- 模型输出（5 种子均值）: prob_mean=0.3339, u_mean=0.6678, r=0.1109
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: direct（直接证据）** — 摘要 21919844 中两药查询词出现在同一句/±150 字符窗口内且含交互/不良反应语境词（('interact', 'advers', 'drug-drug', 'ddi', 'combination', 'concomitant', 'toxicity', 'overdose', 'side effect', 'additive', 'synergis', 'potentiat')）；片段: “Drugs that have a high potential to interact with herbal medicines usually have a narrow therapeutic index, including warfarin, digoxin, cyclosporine, tacrolimus, amitriptyline, midazolam, indinavir, and irinotecan.”
- PubMed 检索结果（retmax=5）:
  - [40331624](https://pubmed.ncbi.nlm.nih.gov/40331624/) — Implementing Pre-Emptive Pharmacogenetics: Impact of Early Pharmacogenetic Screening in a Pediatric Oncology Cohort of 1,151 Subjects.
  - [30361780](https://pubmed.ncbi.nlm.nih.gov/30361780/) — Irinotecan Alters the Disposition of Morphine Via Inhibition of Organic Cation Transporter 1 (OCT1) and 2 (OCT2).
  - [30350190](https://pubmed.ncbi.nlm.nih.gov/30350190/) — Amitriptyline prevents CPT-11-induced early-onset diarrhea and colonic apoptosis without reducing overall gastrointestinal damage in a rat model of mucositis.
  - [22292789](https://pubmed.ncbi.nlm.nih.gov/22292789/) — Herb-drug interactions and mechanistic and clinical considerations.
  - [21919844](https://pubmed.ncbi.nlm.nih.gov/21919844/) — Clinical herbal interactions with conventional drugs: from molecules to maladies.
  - Direct 证据片段（PMID 21919844）: “Drugs that have a high potential to interact with herbal medicines usually have a narrow therapeutic index, including warfarin, digoxin, cyclosporine, tacrolimus, amitriptyline, midazolam, indinavir, and irinotecan.”

## Methantheline + Raltegravir（an increase in the absorption resulting in an increased serum concentration and potentially a worsening of adverse effects cause）
- 药对: `DB00940` / `DB06817`（DrugBank id）; a_name=Methantheline, b_name=Raltegravir
- 事件: an increase in the absorption resulting in an increased serum concentration and potentially a worsening of adverse effects cause
- 模型输出（5 种子均值）: prob_mean=0.3438, u_mean=0.6877, r=0.1074
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Aldosterone + Danazol（The risk or severity of fluid retention increase）
- 药对: `DB04630` / `DB01406`（DrugBank id）; a_name=Aldosterone, b_name=Danazol
- 事件: The risk or severity of fluid retention increase
- 模型输出（5 种子均值）: prob_mean=0.3465, u_mean=0.6931, r=0.1064
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — 检索到 3 篇，其中 2 篇有摘要；均未通过 Direct 判定，也未见药名+机制词共现 → Not identified
- PubMed 检索结果（retmax=5）:
  - [22128732](https://pubmed.ncbi.nlm.nih.gov/22128732/) — [Determination of 12 steroid hormone residues in pig tissues by liquid chromatography-tandem mass spectrometry combining with library search].
  - [3235459](https://pubmed.ncbi.nlm.nih.gov/3235459/) — Ceramic systems for long-term delivery of chemicals and biologicals.
  - [6374601](https://pubmed.ncbi.nlm.nih.gov/6374601/) — [Endocrine dysfunctions in newborn infants during the period of adaptation].
  - （未检出 direct / class-level 证据）

## Aldosterone + (S)-Indapamide（the hypokalemic activities increase）
- 药对: `DB04630` / `DB00808`（DrugBank id）; a_name=Aldosterone, b_name=(S)-Indapamide
- 事件: the hypokalemic activities increase
- 模型输出（5 种子均值）: prob_mean=0.3608, u_mean=0.7215, r=0.1005
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Dexketoprofen + Cyclosporine（The risk or severity of renal failure and hypertension increase）
- 药对: `DB09214` / `DB00091`（DrugBank id）; a_name=Dexketoprofen, b_name=Cyclosporine
- 事件: The risk or severity of renal failure and hypertension increase
- 模型输出（5 种子均值）: prob_mean=0.3615, u_mean=0.7229, r=0.1001
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Zimelidine + Desmopressin（The risk or severity of hyponatremia increase）
- 药对: `DB04832` / `DB00035`（DrugBank id）; a_name=Zimelidine, b_name=Desmopressin
- 事件: The risk or severity of hyponatremia increase
- 模型输出（5 种子均值）: prob_mean=0.5748, u_mean=0.8503, r=0.0861
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Thioridazine + Naltrexone（The risk or severity of hypotension and CNS depression increase）
- 药对: `DB00679` / `DB00704`（DrugBank id）; a_name=Thioridazine, b_name=Naltrexone
- 事件: The risk or severity of hypotension and CNS depression increase
- 模型输出（5 种子均值）: prob_mean=0.4043, u_mean=0.8086, r=0.0774
- semantic_overlap: yes（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）
- **evidence_auto: class_suggested（类别级机制建议，需人工裁决）** — 摘要 19393386 含至少一个药名且命中事件机制词 ['sedat']；命中句片段见下（供人工裁决）
- PubMed 检索结果（retmax=5）:
  - [19393386](https://pubmed.ncbi.nlm.nih.gov/19393386/) — [Treatment of a serious autistic disorder in a child with Naltrexone in an oral suspension form].
  - [7603392](https://pubmed.ncbi.nlm.nih.gov/7603392/) — Naltrexone for alcohol dependence.
  - [7904217](https://pubmed.ncbi.nlm.nih.gov/7904217/) — Efficacy of psychotropic drugs for reducing self-injurious behavior in the developmental disabilities.
  - [2912237](https://pubmed.ncbi.nlm.nih.gov/2912237/) — More on idiosyncratic reaction to naltrexone.
  - [3369583](https://pubmed.ncbi.nlm.nih.gov/3369583/) — Idiosyncratic reaction to naltrexone augmented by thioridazine.
  - Class-level 命中句 1/2（PMID 19393386，机制词 ['sedat']）: “Certain side effects were observed, namely transitory sedation at the beginning of treatment and moderate constipation.”

## Mometasone furoate + Bendroflumethiazide（The risk or severity of electrolyte imbalance increase）
- 药对: `DB14512` / `DB00436`（DrugBank id）; a_name=Mometasone furoate, b_name=Bendroflumethiazide
- 事件: The risk or severity of electrolyte imbalance increase
- 模型输出（5 种子均值）: prob_mean=0.4108, u_mean=0.8216, r=0.0733
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）

## Revefenacin + Topiramate（The risk or severity of hyperthermia and oligohydrosis increase）
- 药对: `DB11855` / `DB00273`（DrugBank id）; a_name=Revefenacin, b_name=Topiramate
- 事件: The risk or severity of hyperthermia and oligohydrosis increase
- 模型输出（5 种子均值）: prob_mean=0.4518, u_mean=0.9037, r=0.0435
- semantic_overlap: no（事件为 10 个语义不相交事件之一）
- **evidence_auto: none（未识别）** — PubMed 检索无结果（两药名联合查询无命中）
- PubMed 检索结果（retmax=5）:
  - （无检索结果）
  - （未检出 direct / class-level 证据）
