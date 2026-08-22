# 外部验证章节重做设计：RxPairEvid-50K（审稿回应）

- 日期：2026-08-21
- 论文：`C:\Users\Admin\Desktop\fyx8_21.tex`（UAID-DDI / PharDDIE + EviDDIE）
- 仓库：`D:\PharDDIE and EviDDIE\PharDDIE_github_8_10`
- 状态：已确认设计，待实施

## 1. 背景与目标

**审稿意见**（大意）：数据集重叠，要求换新数据集验证并重做实验。

**现状**：主实验仅用 DrugBank 衍生的 Dataset 1（Nyamabo et al.，1,706 药 / 86 事件 / 191,808 记录）；Case Study 的 8 个案例是"DrugBank 内部一致性检查"（标签与验证同源），论文 Limitations 也自认"缺少外部验证"。

**已确认的决策**（与用户逐项确认）：

| 决策点 | 结论 |
|---|---|
| 应对方式 | 保留 Dataset 1 结果，**新增外部验证章节**（不整体替换主实验） |
| 验证模型 | **PharDDIE 为主**（1-shot/5-shot 五种子）；找不到 checkpoint 则**兜底改用 EviDDIE 零样本**（五种子模型完好，在 `EviDDIE/models/dataset1/`） |
| 新数据集 | **RxPairEvid-50K**（FAERS 药物警戒来源，2025-11 发布 / 2026 论文，Mendeley DOI 10.17632/zrvzpfmzcz.1，license-clean，自带 SHA-256） |
| 是否重训 | **不重训**（用户明确决定）。PharDDIE 五种子 checkpoint 本机/GitHub 均缺失，由用户在外查找 |
| 时间预算 | 2-3 周内交修改稿 |
| 约束 | 不改动仓库任何现有文件；GitHub 只读不推送；新增文件集中在 `external/` 目录 |

**Checkpoint 查找截止点**：2026-08-24（3 天）。到期未找到 → 自动转入 EviDDIE 兜底分支，管线不变，仅换推理入口与事件原型来源。

## 2. 总体架构

全部新增文件放在 `external/`，不改动仓库现有文件；新数据集按 dataset1 布局建 `PharDDIE/dataset_ext/`（EviDDIE 分支加 `EviDDIE/dataset_ext/`）。

```
external/
├── raw/                         # 用户手动下载的 RxPairEvid-50K 原始文件
│   ├── ddi_pairs_50k.csv
│   ├── codebook.md
│   ├── checksums.txt
│   └── provenance.md
├── fetch_rxpairevid.ps1         # checksums 校验（下载由用户手动完成）
├── rxpairevid_to_dataset.py     # IK14→SMILES(PubChem)、事件分层、构建 dataset_ext
├── audit_overlap_ext.py         # 与 Dataset 1 药物重叠审计（回应核心证据）
├── pharddie_export_ext.py       # 主路径：PharDDIE 五种子新数据推理
├── eviddie_export_ext.py        # 兜底路径：EviDDIE 零样本新事件推理
├── ext_summary_table.py         # 外部验证汇总表（分层 AUROC/AUPRC/ACC/F1 + 基线）
├── case_study_ext.py            # 客观案例选取 + 独立文献佐证表
└── outputs/                     # 全部产物（表、图、审计记录）
```

依赖复用：`shared/neg_manifest.py`（负样本 manifest 协议）、`shared/audit_logger.py`（均以参数调用，不改动）。RDKit 药物图特征逻辑**在 external 脚本内自含一份**（复制自 `shared/preprocess.py` 的药物图构建部分并参数化数据集路径；原文件 `shared/preprocess.py` 保持零改动，因为它硬编码了 dataset1 路径）。

## 3. 数据流（RxPairEvid-50K → 事件 episode）

1. **校验**：`fetch_rxpairevid.ps1` 对 `checksums.txt` 逐文件 SHA-256 校验，失败即中止。
2. **药物映射**：提取全部 IK14 → PubChem PUG REST identifier exchange（InChIKey→CID→canonical SMILES）；映射失败剔除并写入 `outputs/mapping_audit.csv`；映射率 <70% 时启用 CIR/UniChem 备用源。
3. **重叠审计**：Dataset 1 的 1,706 个 SMILES 计算 IK14，与 RxPairEvid 药集求交集，输出 `outputs/drug_overlap_report.csv`（重叠数量/比例、双向清单）。此数字如实报告（预期低），作为回应信第一张证据。
4. **事件分层**（2026-08-21 实测数据修正）：`faers_best_pt_code_strict`（MedDRA PT code）作为每药对的主事件键。实测：50K 行中阳性信号对 873 个（1.75%，其余 49,127 为无信号对），横跨 **429 个 PT**；每 PT 阳性对数 min 1 / median 1 / max 13。因此分层方案调整为：**1-shot 层 = 阳性对数 ≥2 的全部 PT（185 个事件）；5-shot 层 = 阳性对数 ≥6 的 PT（24 个事件）**。此规模与论文原 held-out 集（13 fewer + 10 rare 事件）量级相当，无需 common 层。事件总数、剔除数全部记录。
5. **episode 构建**：复用仓库协议——负采样两套并报：**(i) tail-corruption**（1:1，与主实验协议可比，为主要表）；**(ii) 原生负样本**（从 49,127 个无信号对中按事件 1:1 采样，反映真实筛查阳性率，为补充表）。每事件固定种子 manifest（5 种子 × 1/5-shot，与 `neg_manifests` 同构）、support/query 隔离断言、SHA256 记录。
6. **DRKG 兜底**：`ent2embids` 缺失药物置 -1 → 随机向量兜底（现有代码路径）；无 KG 邻居走 ACI 结构分支；输出兜底药物比例（本身即回应素材：新药上模型不依赖 KG）。
7. **推理**：
   - 主路径：`pharddie_export_ext.py`（克隆 `pharddie_export_full.py`）加载 5 种子 checkpoint，对 dataset_ext 各层 episode 推理，输出每样本概率 + 不确定度 + 4 列溯源哈希（checkpoint/manifest/embedding/git）。
   - 兜底路径：`eviddie_export_ext.py`（克隆 `eviddie_export_zs_v2.py`）对 dataset_ext 的新事件（PT code）推理；**事件原型来源**：PT code → 文本映射需 MedDRA PT 标签（UMLS/BioPortal 机构授权或 PubMed 描述），映射失败的事件剔除并记录（见 §5 风险 ④）。
   - 基线：RareDDIE 五种子（本机 `C:\Users\Admin\UAID-DDI\PharDDIE\models\`）同 episode 对比（仅主路径需要）。
8. **案例研究**：`case_study_ext.py` 按 triage 策略 `r = p(1-u)` 排名，从 rare 层取 **top-10** 候选（与论文原案例的 top-10 体量一致）→ 排除与 Dataset 1 重叠药对 → 每例证据用 PubMed 文献 PMID + FAERS 信号统计独立佐证（**禁止使用 DrugBank 记录作为证据**）→ 输出案例表（Rank / Drug pair / 事件 / Prob. / Uncertainty / Evidence 来源）。案例来源：主路径 = PharDDIE 1-shot 五种子均值排名；兜底路径 = EviDDIE 0-shot 五种子均值排名。

## 4. 指标与输出

- 分层（common / rare）与总体：AUROC、AUPRC、ACC、event-macro F1，5 种子 mean±SD。
- 主路径额外：RareDDIE 同协议对比行 + 种子配对差异（复用 `shared/paired_diff_rareddie.py` 思路）。
- 兜底路径额外：no-skill 参考行（复用 `shared/calibration_table.py` 思路）。
- 案例表 ≥8 例，证据全部来自独立来源（PMID / FAERS）。

## 5. 错误处理与风险预案

| 风险 | 预案 |
|---|---|
| ① Mendeley 下载受限 | 用户手动下载放入 `external/raw/`，脚本只做校验 |
| ② PubChem 映射率 <70% | CIR / UniChem 备用源；仍不足则报告并缩小评估集 |
| ③ 某 PT 阳性对不足 | 实测已确认：1-shot 层 185 事件、5-shot 层 24 事件（≥6 对）；不足者剔除并报告 |
| ④ MedDRA 无 PT 文本（EviDDIE 分支） | UMLS/BioPortal（机构授权）或 PubMed 描述提取；失败事件剔除并记录 |
| ⑤ checkpoint 3 天内未找到 | 2026-08-24 起自动转入 EviDDIE 兜底分支 |
| ⑥ 推理中途 OOM | 分事件层分块推理，逐块落盘续跑 |

## 6. 论文改动清单（fyx8_21.tex）

1. Results 新增小节 "External Validation on Pharmacovigilance-Derived Data"（表 + 重叠审计数字 + 兜底药物比例）。
2. Case Study 小节重写（新案例表，证据列改为 PMID/FAERS；删除"内部一致性"表述）。
3. Limitations 修订（删除"缺少外部验证"的自认，改写为"已在一组独立药物警戒数据上验证"并保留剩余局限）。
4. Data and Code availability 补充 RxPairEvid 引用（DOI 10.17632/zrvzpfmzcz.1）与 `external/` 脚本说明。
5. 审稿回应信初稿（点对点：重叠质疑 → 数据来源独立性 + 药物重叠审计数字 + 新数据结论）。

## 7. 验收标准

- [ ] 药物重叠审计报告产出且数字可写进回应信
- [ ] 外部验证汇总表（分层 + 总体 + 基线）
- [ ] 新案例表 ≥8 例，证据全部独立来源；**top-10 中 ≥7 例获独立证据证实（用户明确成功标准，2026-08-21）**
- [ ] 论文 4 处改动 + 回应信初稿完成
- [ ] 一条命令可复现（`external/` 内 fail-fast 流水线，含 SHA256 校验）
- [ ] 全部产物进 `external/outputs/` 且 SHA256 记录

## 8. 时间线（2026-08-21 起）

| 天数 | 任务 | 依赖 |
|---|---|---|
| Day 0-1 | 用户：下载数据集 + 找 checkpoint；我：脚本骨架 + 校验脚本 | 无 |
| Day 1-3 | 数据管线 + 重叠审计 | 数据集到位 |
| Day 3 | checkpoint 决策点（找到→主路径；未找到→EviDDIE 分支） | 截止日 |
| Day 3-5 | episode 构建 + 五种子推理 + 基线 | 数据管线 + 模型 |
| Day 5-8 | 案例研究 + 指标表 | 推理结果 |
| Day 8-11 | 论文改写 + 回应信 | 全部产物 |
| Day 11-14 | 缓冲 + 证据链自审 + 可复现性复跑 | — |

## 9. 待用户提供的两项输入

1. **RxPairEvid-50K 原始文件**（Mendeley DOI 10.17632/zrvzpfmzcz.1，需登录下载）：`ddi_pairs_50k.csv`、`codebook.md`、`checksums.txt`、`provenance.md` → 放入 `external/raw/`。
2. **PharDDIE 五种子 checkpoint**（若找到）：`models_drugbank_1shot_str_seed{19940419,20230801,20240115,20240520,20240910}/bestmodel` 与 `models_drugbank_5shot_str_seed{...}/bestmodel` → 放入 `PharDDIE/models/dataset1/` 对应路径（或告知原路径）。注意：本地训练日志中 5-shot 仅存 4 个种子（缺 seed19940419 的日志），该跑位若从未保存过模型，5-shot 将以 4 种子报告并在论文中注明。
