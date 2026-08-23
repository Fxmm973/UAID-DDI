# Dataset 2 外部验证与案例研究 — 复现手册

本目录（`external/`）包含 EviDDIE 在 Dataset 2（Lin 协议独立基准）上重训、外部验证与案例研究的完整可复现管线。全部脚本已提交；大文件（BioSentVec 模型 21GB、DRKG npy、FAERS 原始包等）按 `.gitignore` 排除，可按下列命令重新获取或从 dataset1 复制。

## 环境

- Python: `C:\Users\Admin\.conda\envs\PharDDIE\python.exe`（torch 2.0.1+cu118, PyG 2.6.1, RDKit, pandas, numpy, scikit-learn）
- GPU: NVIDIA RTX 4090（CPU 也可跑推理，速度较慢）

## 数据与模型准备（一次性）

1. Dataset 2（已入库小文件；两个大文件从 dataset1 复制）：
   ```bash
   cp EviDDIE/dataset1/DRKG_TransE_entity.npy EviDDIE/dataset2/
   cp EviDDIE/dataset1/DRKG_TransE_relation.npy EviDDIE/dataset2/
   cp PharDDIE/dataset1/path_graph_train_only EviDDIE/dataset2/
   ```
2. BioSentVec 模型（推理原型与训练语义文件用，21GB，不入库）：
   ```bash
   # NCBI 官方：https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin
   # 下载到 external/biosentvec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin（22,475,736,490 字节）
   # 校验：加载器自验（external/biosentvec_loader.py --validate 用 EviDDIE/dataset1/event_embedding2.json 的 92 向量，mean cos 需 =1.0）
   ```

## 训练（五种子，~3.5h/种子）

```bash
PY= C:/Users/Admin/.conda/envs/PharDDIE/python.exe   # 填实际路径
for s in 19940419 20230801 20240520 20260201 20260301; do
  $PY external/train_eviddie_dataset2.py --seed $s --max-batches 20000 &
done
# 注：论文五种子中 20240115/20240910 在 Dataset 2 上确定性坍缩（loss 恒 1.3333），
# 按预注册规则替换为 20260101→又坍缩→20260201/20260301（详见训练日志与报告）；
# 坍缩种子的替换已在训练报告（task-13-report.md）中披露。
```

## 推理（held-out test2，25 事件）

```bash
$PY external/eviddie_export_ds2.py \
    --seeds 19940419,20230801,20240520,20260201,20260301
# -> external/outputs/predictions_ds2_retrained_0shot.csv (18,720 行)
```

## 汇总与审计

```bash
# 汇总表（含完整层与 cos<0.7 无重叠子集层）
$PY external/ext_summary_dataset2.py        # 若 T12 脚本名不同以实际为准
# 事件语义重叠（Dataset 1 vs 2，BioSentVec 余弦）
# 判定文件：external/outputs/disjoint_events.json（10 个无语义对应事件）
# 案例泄漏审计（案例候选 vs train/dev，必须 PASS）
$PY external/audit_case_leakage.py
# -> external/outputs/case_leakage_audit.json: VERDICT PASS, 0/25 pair/triple hits
```

## 案例研究（每事件取 1，25 行全量报告）

```bash
$PY external/case_study_per_event.py
# -> case_candidates_dataset2_per_event_v2.csv（25 行，含 evidence_auto 三档）
# -> case_evidence_dataset2_v2.md（逐条证据片段，全部 PMID 经真实标题核验）
```

## 关键结果速查

- 泛化边界（AUC, 5 种子 mean±SD）：零样本跨基准 0.494±0.023；重训后无重叠子集 0.4984±0.0213；重训后完整 test2 0.5718±0.0125；基准内 rare 0.712（论文自带 CSV）。
- 三层重叠审计：KG 边 0/1840；药物级 78.9%；药对级 58.0%；事件语义对应 58/80（0 文本精确匹配）。
- 案例证据三档：direct 0 / class-level 15 / none 10（class-level = 单药机制文献支撑，作者裁决）。
- 案例泄漏：PASS（0/25）。

## SHA256 证据链

- 预测 CSV 每行含 checkpoint_sha256 / eval_manifest_sha256 / event_embedding_sha256 / git_commit。
- 负样本 manifest 哈希：EviDDIE/dataset2/neg_manifests/manifest_hashes.json。
- 训练 checkpoint 哈希与 dev manifest：训练日志目录 external/outputs/train_logs_ds2/ 的 bestmodel_meta.json。
