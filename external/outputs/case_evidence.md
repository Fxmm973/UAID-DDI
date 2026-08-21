# Case-Study Evidence: PubMed 独立文献佐证（人工复核材料）

- 生成日期: 2026-08-21
- 来源预测 CSV: `predictions_rxpairevid_eviddie_1shot_0shot.csv`（1-shot tier, y_true=1）
- 选择规则: 每 (drug_a, drug_b, event) 在 5 个训练种子上取 prob/uncertainty 均值；r = p_mean·(1−u_mean)；剔除与 Dataset 1 药对级重叠的 304 个信号对 (R15-ext)；按 r 降序取 top-10。
- **启发式佐证计数: 0/10**（标题同时提及两药且含交互/不良反应语境关键词；仅作人工复核提示，非最终科学判断；R12 目标 ≥7/10；FAERS 信号统计不视为独立证据）
- 检索式: `"a_name"[All Fields] AND "b_name"[All Fields] AND (interaction OR adverse)`，PubMed Entrez，top-3 PMID+标题，0.4s/请求限速。

## 证据列说明（论文表格用）
- 独立证据列：PubMed PMID（唯一独立佐证来源，R12）
- 非独立证据列：FAERS 统计（标签源自 FAERS，仅作背景摘录，不计入佐证计数）

## Rank 1 — Semaglutide + Hydrochlorothiazide（PT-18561528）
- 药对: `DLSWIYLPEUIQAV` / `JZUFKLXOESDKRF`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4240, u_mean=0.8304, r=0.0719
- FAERS（非独立证据，仅摘录）: n_reports=139, PRR_max_strict=66829.15238095238, ROR95_lcl_max_strict=23051.348071574947
- 佐证标志: NO
- PubMed 检索结果:
  1. [36034061](https://pubmed.ncbi.nlm.nih.gov/36034061/) — Fixed Drug Eruption: An Underrecognized Cutaneous Manifestation of a Drug Reaction in the Primary Care Setting.

## Rank 2 — Hydrocortisone acetate + Celecoxib（PT-21612999）
- 药对: `ALEXXDVDDISNDU` / `RZEKVGVHFLEQIL`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4239, u_mean=0.8305, r=0.0719
- FAERS（非独立证据，仅摘录）: n_reports=3243, PRR_max_strict=28669.29839704069, ROR95_lcl_max_strict=1685.009926572578
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 3 — Triamcinolone + Clozapine（PT-16599628）
- 药对: `GFNANZIMVAIWHM` / `QZUDBNBUXVUHMW`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4241, u_mean=0.8308, r=0.0718
- FAERS（非独立证据，仅摘录）: n_reports=19, PRR_max_strict=320085.13333333336, ROR95_lcl_max_strict=127308.36137631266
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 4 — Cortisone acetate + Ephedrine（PT-20937945）
- 药对: `ITRJWOMZKQRYTA` / `KWGRBVOPPLSCSI`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4241, u_mean=0.8309, r=0.0717
- FAERS（非独立证据，仅摘录）: n_reports=3276, PRR_max_strict=28380.381751602075, ROR95_lcl_max_strict=1667.9781182112786
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 5 — Lactulose + (R)-Fluoxetine（PT-18265723）
- 药对: `JCQLYHFGKNRPGE` / `RTHCYVBBDHJXIQ`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4248, u_mean=0.8320, r=0.0714
- FAERS（非独立证据，仅摘录）: n_reports=376, PRR_max_strict=19591.472148541117, ROR95_lcl_max_strict=6184.398473574445
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 6 — Topiramate + Amphetamine（PT-18339940）
- 药对: `KJADKKWYZYXHBB` / `KWTSXDURSIMDCE`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4265, u_mean=0.8355, r=0.0701
- FAERS（非独立证据，仅摘录）: n_reports=219, PRR_max_strict=54677.41038961039, ROR95_lcl_max_strict=16653.42338432871
- 佐证标志: NO
- PubMed 检索结果:
  1. [42234418](https://pubmed.ncbi.nlm.nih.gov/42234418/) — Evaluating Reduced Use and Abstinence as Outcomes in Pharmacotherapy Trials for Stimulant Use Disorder: A Meta-Analysis of 12 Randomized Controlled Trials.
  2. [38988470](https://pubmed.ncbi.nlm.nih.gov/38988470/) — Efficacy of Pharmacotherapies for Bulimia Nervosa: A Systematic Review and Meta-Analysis.
  3. [38566910](https://pubmed.ncbi.nlm.nih.gov/38566910/) — Exploring the association between weight loss-inducing medications and multiple sclerosis: insights from the FDA adverse event reporting system database.

## Rank 7 — Methotrexate + Quetiapine（PT-8859257）
- 药对: `FBOZXECLQNJBKD` / `URKOMYMAXPYINW`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4277, u_mean=0.8375, r=0.0695
- FAERS（非独立证据，仅摘录）: n_reports=3166, PRR_max_strict=37757.38143353332, ROR95_lcl_max_strict=2253.566682069588
- 佐证标志: NO
- PubMed 检索结果:
  1. [34798832](https://pubmed.ncbi.nlm.nih.gov/34798832/) — Prevalence of potentially harmful multidrug interactions on medication lists of elderly ambulatory patients.

## Rank 8 — 2-chloro-5-[(1S)-1-hydroxy-3-oxo-2H-isoindol-1-yl]benzenesulfonamide + Dimenhydrinate（PT-18517210）
- 药对: `JIVPVXMEBJLZRO` / `ZZVUWRFHKOJYTH`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4276, u_mean=0.8376, r=0.0694
- FAERS（非独立证据，仅摘录）: n_reports=395, PRR_max_strict=279770.95959595963, ROR95_lcl_max_strict=17073.40667626671
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 9 — Perindopril + Indium In-111 oxyquinoline（PT-18813849）
- 药对: `IPVQLZZIHOAWMC` / `MCJGNVYPOGVAJF`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4278, u_mean=0.8381, r=0.0693
- FAERS（非独立证据，仅摘录）: n_reports=13, PRR_max_strict=189940.88571428572, ROR95_lcl_max_strict=73992.02177746226
- 佐证标志: NO
- PubMed 检索结果:
  - （无检索结果）

## Rank 10 — Budesonide + Metformin（PT-18358513）
- 药对: `VOVIALXJUBGFJZ` / `XZWYZXLIPXDOLR`（IK14）
- 模型输出（5 种子均值）: prob_mean=0.4275, u_mean=0.8381, r=0.0692
- FAERS（非独立证据，仅摘录）: n_reports=2173, PRR_max_strict=46865.34590616376, ROR95_lcl_max_strict=2775.395128376972
- 佐证标志: NO
- PubMed 检索结果:
  1. [41821050](https://pubmed.ncbi.nlm.nih.gov/41821050/) — Context-dependent effect of glucocorticoid receptor activity shapes ovarian cancer cell plasticity and therapy response.
  2. [40607383](https://pubmed.ncbi.nlm.nih.gov/40607383/) — ER stress genes (COL1A1, LOXL2, VWF) predicts IKK-16 as a Candidate therapeutic target for colitis-related inflammation and fibrosis suppression.
  3. [20158286](https://pubmed.ncbi.nlm.nih.gov/20158286/) — Prospective drug safety monitoring using the UK primary-care General Practice Research Database: theoretical framework, feasibility analysis and extrapolation to future scenarios.
