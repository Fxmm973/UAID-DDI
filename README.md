# UAID-DDI: A Reliability-Oriented Framework for Few-Shot and Zero-Shot Drug–Drug Interaction Event Prediction with Selective Referral

Official implementation of the paper *"A Reliability-Oriented Framework for Few-Shot and Zero-Shot Drug–Drug Interaction Event Prediction with Selective Referral"*.

## Overview

UAID-DDI is a two-model framework for predicting rare drug-drug interaction (DDI) events under extreme data scarcity. Each model outputs a prediction probability $p$ together with an uncertainty signal, which drives a downstream **selective-referral evaluation using confidence or model-derived uncertainty signals**. Three uncertainty signals are distinguished throughout the paper and code:

| Signal | Setting | Meaning |
|--------|---------|---------|
| $u_{\text{entropy}} = H(p)$ | PharDDIE 1-shot | prediction entropy; a confidence-derived baseline (NOT epistemic uncertainty) |
| $u_{\text{latent}}$ | PharDDIE $K\ge 5$ | normalized SRAE **latent dispersion score**; a reconstruction-derived proxy with no KL-based posterior interpretation |
| $u_{\text{EDL}} = 2/S$ | EviDDIE zero-shot | Dirichlet evidential uncertainty (total evidence $S$) |

The selective-referral policy maps each candidate to one of four actions---High-Priority Review, Expert Referral, Deferred Review, Low Priority---under the **unified semantics**: automatic set = {high-priority review, low-priority assignment} ($u \le \tau_u$); referred set = {expert referral, deferred review} ($u > \tau_u$). The paper reports selective-referral results for the 1-shot setting; the rebuilt per-signal selective-referral comparison (raw score $p$ / MSP / margin / entropy / latent $u$ / true random referral) is reported at matched coverage and at fixed referral budgets.

| Model | Setting | Key Modules |
|-------|---------|-------------|
| **PharDDIE** | Few-Shot ($K \in \{1,5\}$) | SHCR (Selected Hidden-Channel Reweighting) + ACI (Adaptive Context Integration) + SRAE (Stochastic Reconstruction-Regularized Autoencoder) |
| **EviDDIE** | Zero-Shot | BSA (Bio-Semantic Alignment) + EVI (Evidential Inference: standard two-class EDL, native dual-output evidential head) |

### PharDDIE — Few-Shot DDI Prediction

- **SHCR** (Selected Hidden-Channel Reweighting): `pharddie_layers.py` — `HiddenChannelReweightingTransformerConv` fuses a learned data-driven gate $\psi_i$ with a fixed-channel proxy signal $\phi_i$ at a fixed 0.7:0.3 ratio. Five fixed channel indices (0, 1, 2, 46, 53) of the *projected hidden* representation (after `initial_node_feature` linear projection + LayerNorm + ELU) are reweighted by learnable channel coefficients; the indices carry no guaranteed chemical meaning and the module acts as a lightweight regularizing prior, **not** a chemical detector. Node scaling: $\tilde{h}_i = h_i \odot (1 + 1.5\gamma_i)$.
- **ACI** (Adaptive Context Integration): `pharddie_matcher.py` — bilinear attention over first-order DRKG neighbors with differential query $\delta_{ij} = z_j - z_i$ and residual-style gating $d_i = \text{ELU}(W_{\text{nei}} c_i + W_{\text{self}} z_i)$; drugs without KG neighbors fall back to the structural branch.
- **SRAE** (Stochastic Reconstruction-Regularized Autoencoder): `pharddie_matcher.py` — asymmetric stochastic bottleneck ($\eta = 10^{-2}$ support, $\eta = 10^{-3}$ query), no KL term and no standard-normal prior; the scale output $\sigma_\phi = \exp(0.5\, l_\phi)$ is a learned noise magnitude, reported as the *latent dispersion score*. Pair scoring: $\text{MLP}(|z_s - z_q|)$.

### EviDDIE — Zero-Shot DDI Prediction

- **BSA** (Bio-Semantic Alignment): `eviddie_matcher.py` — a GAN aligns drug-pair latent codes with BioSentVec event prototypes (700-dim, precomputed in `event_embedding2.json`). Generator: $700\to256\to512\to64$ (Tanh). Critic: $64\to512\to256\to128\to1$ (Sigmoid).
- **EVI** (Evidential Inference): `eviddie_matcher.py` — native dual-output Dirichlet evidential head ($\alpha = e + 1$, $u_{\text{EDL}} = 2/S$), EDL loss with annealed KL ($\lambda_t = \min(1, t/10000)$). **Comparator**: $\mathrm{Softplus}(\mathrm{MLP}(|p_t - z_q|))$, the absolute-difference comparator described in the paper (legacy concatenation-based comparators and single-output checkpoints are rejected, never converted).
- The formal training entry `eviddie_trainer.py` uses per-seed independent checkpoints (`models/{prefix}_seed{seed}bestmodel{, _G}`) for the five training seeds (19940419, 20230801, 20240115, 20240520, 20240910). Inference uses the raw BioSentVec prototypes (no semantic noise) and one fixed evaluation manifest (seed 19940419).

### EviDDIE Frozen-Backbone Ablation

`eviddie_train_ablation.py` trains comparator heads **from scratch on the frozen five-seed backbones** (KG neighbor encoder, MVN_DDI, SRAE, GAN generator all frozen). Four heads are trained for 5,000 iterations per variant under the internal dev protocol:

| Variant | Head / loss |
|---------|-------------|
| `softmax` | cross-entropy comparator head |
| `evi_no_evi` | evidential head, MSE **without** the EVI KL regularizer |
| `wo_BSA` | evidential head with a trainable linear prototype projection replacing the BSA GAN generator |
| `evi_full` | the native evidential head trained with the complete EDL loss (MSE + annealed KL) |

The production checkpoint (jointly trained head + backbone) serves as a horizontal reference evaluated under the identical protocol. Results, paired $t$-tests, and figures are produced by `eviddie_ablation_summary.py`, `eviddie_ablation_sigtest.py`, `eviddie_ablation_figure.py`, and `eviddie_ablation_curves_figure.py`.

---

## Repository Structure

```
UAID-DDI/
├── environment.yml                 # Conda environment (PyTorch 2.0.1+cu118, PyG 2.6.1, RDKit 2025.03.5)
├── reproduce.ps1                   # Fail-fast pipeline: manifest verification -> leakage audit -> exports -> tables (no training)
├── shared/                         # Shared utilities & paper-table generators
│   ├── preprocess.py               # Molecular featurization (RDKit atom/bond features)
│   ├── checkpoint.py               # Safe checkpoint loading with audit logging
│   ├── neg_manifest.py             # Negative-sample manifest generation (SHA256 audited)
│   ├── verify_manifests.py         # SHA256 + entry-count verification of all manifests
│   ├── eval_manifest.py            # Fixed evaluation-manifest helpers
│   ├── audit_leakage.py            # Six-part leakage audit (support-query / pos-neg / ordered / unordered / cross-split / KG-edge)
│   ├── audit_drug_overlap.py       # Drug-overlap audit across splits
│   ├── audit_logger.py             # Audit trail utilities
│   ├── calibration_table.py        # P0-2/P0-5 Table: AUROC/AUPRC/ACC/F1/Brier/NLL/ECE/HCE (+TempScale, no-skill row, reliability diagram with per-bin counts); HCE = classification error among confidence>=0.9
│   ├── rq3_selective_referral.py   # P0-3 rebuilt per-signal selective referral (p / MSP / margin / entropy / latent u / random; selective risk = classification error rate; per-seed AURC mean±SD; error-detection AUROC/AUPRC)
│   ├── rq3_triage_table.py         # Selective-referral / confidence-based prioritization table (unified semantics; includes the independent Random row with correct error-rate risk)
│   └── paired_diff_rareddie.py     # P1-6: seed-paired PharDDIE-RareDDIE differences with 95% CI
│
├── PharDDIE/                       # Few-shot model
│   ├── pharddie_args.py            # Hyperparameters & CLI
│   ├── pharddie_dataloader.py      # Episodic data loading
│   ├── pharddie_layers.py          # SHCR: HiddenChannelReweightingTransformerConv
│   ├── pharddie_models.py          # MVN_DDI: molecular encoder with SAGPooling
│   ├── pharddie_matcher.py         # EmbedMatcher: ACI + SRAE + scorer
│   ├── pharddie_trainer.py         # Training loop; dev checkpoint selection under the fixed-seed dynamic validation protocol
│   ├── pharddie_train_wo_unc.py    # Ablation: train without uncertainty branch
│   ├── pharddie_export.py          # w/o-uncertainty variant export (manifest-based)
│   ├── pharddie_export_full.py     # Main export: fixed manifests, SHA256-verified, SEED-CHAIN-checked
│   ├── pharddie_table2.py          # Paper Table 2 (main results; 7 transcribed baselines + re-evaluated RareDDIE)
│   ├── pharddie_table3.py / pharddie_table3_complete.py  # PharDDIE calibration rows (per-seed aggregation)
│   ├── pharddie_table4_paper.py    # Legacy Table 4 generator (unified semantics; superseded by shared/rq3_*.py for the rebuilt per-signal tables)
│   ├── eval_rareddie_unified.py / aggregate_rareddie.py  # RareDDIE re-evaluation under the unified protocol
│   ├── dataset1/                   # Benchmark dataset (few-shot split) + neg_manifests/ (SHA256-recorded)
│   └── results/                    # Table outputs + per-seed RareDDIE results + rq3_rebuilt_PharDDIE.csv (source of Tables 5/6)
│
├── EviDDIE/                        # Zero-shot model
│   ├── eviddie_args.py             # Hyperparameters & CLI
│   ├── eviddie_dataloader.py       # Data loading
│   ├── eviddie_models.py           # Molecular encoder (standard TransformerConv)
│   ├── eviddie_matcher.py          # Matcher with BSA (GAN) + EVI (native dual-output EDL, |p_t − z_q| comparator)
│   ├── eviddie_trainer.py          # Formal training entry (per-seed checkpoints, 5 seeds)
│   ├── eviddie_train_ablation.py   # Frozen-backbone 4-variant head ablation (softmax / w/o EVI / w/o BSA / full EDL)
│   ├── eviddie_export_zs_v2.py     # Zero-shot export (fixed manifest, per-seed checkpoints, per-sample CSV with 4 hash columns)
│   ├── eviddie_ablation_summary.py # Ablation metric aggregation (4 metrics × 3 settings × 4 variants)
│   ├── eviddie_ablation_sigtest.py # Paired t-tests vs the complete model (per setting × metric)
│   ├── eviddie_ablation_figure.py  # Main ablation figure (AUROC+F1, significance stars) + 4-metric supplement
│   ├── eviddie_ablation_curves_figure.py  # Training-dynamics curves + production-checkpoint reference
│   ├── eviddie_eval_full_dev.py    # Full-model dev evaluation under the internal dev protocol
│   ├── eviddie_reliability_figure.py  # Horizontal reliability diagram (Fewer | Rare, native + TempScale, per-bin counts)
│   ├── eviddie_table_discrimination.py # Zero-shot discrimination table (legacy CSV reader; superseded by shared/calibration_table.py)
│   ├── eviddie_export_variants.py / eviddie_export_zs.py  # Legacy exports (kept for provenance)
│   ├── neg_manifests/              # Pre-generated negative manifests + SHA256 hashes (all 5 seeds × dev/test/test2)
│   ├── dataset1/                   # Benchmark dataset (incl. event_embedding2.json BioSentVec prototypes)
│   └── results/                    # Ablation curves/summary/sigtest, calibration table, RQ3 zero-shot CSVs, figures
│       └── predictions/            # Per-sample zero-shot predictions (5 seeds, fixed manifest) + episode manifests
│
├── results/                        # Paper-facing table summaries
│   └── table4_paper.txt            # Selective-referral performance / confidence-based prioritization (unified semantics, 1-shot)
│
├── tests/
│   └── test_evidential_class_order.py  # Class-order convention test (negative=0, positive=1)
│
├── configs/                        # Model configs (eviddie.json, pharddie.json)
└── audit/                          # Evidence chain
    ├── checkpoints_sha256.md       # SHA256 manifest of all per-seed checkpoints + BioSentVec embeddings
    ├── environment/                # Environment lock record
    ├── figure_audit_checklist.md   # Figure-audit checklist
    ├── leakage_reports/            # Six leakage-audit reports (all PASS on Dataset 1)
    └── training_logs/              # Per-seed training logs (PharDDIE 1/5-shot, EviDDIE 0-shot)
```

> **Large files not included**: DRKG TransE entity embeddings (`DRKG_TransE_entity.npy`, ~200 MB), Morgan fingerprint features, DrugBank-derived training-task files, and trained model checkpoints (16-33 MB each). Their SHA256 values are recorded in `audit/checkpoints_sha256.md`; the binaries can be regenerated with the provided scripts or obtained from the authors (Zenodo deposit on acceptance).

---

## System Requirements

Tested on Ubuntu 16.04, CentOS 7, and Windows 11 with Python 3.9 on one NVIDIA RTX 4090 GPU (24 GB).

---

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate PharDDIE

# Or install manually
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.6.1 rdkit-pypi==2025.03.5 numpy==1.24.3 pandas scikit-learn tqdm tensorboardX
```

---

## Quick Start

### 1. Data Preparation

The benchmark dataset is pre-processed and included in `PharDDIE/dataset1/` (and mirrored under `EviDDIE/dataset1/`). They were derived from DrugBank (version 5.x, license-restricted) and processed following the event-level split protocols from Nyamabo et al. (Briefings in Bioinformatics, 2022) and Lin et al. (Briefings in Bioinformatics, 2021).

Negative-sample manifests are pre-generated in `PharDDIE/dataset1/neg_manifests/` and `EviDDIE/neg_manifests/` with SHA256 hashes. Verify them with:

```bash
python shared/verify_manifests.py --hash-log PharDDIE/dataset1/neg_manifests/manifest_hashes.json --manifest-dir PharDDIE/dataset1/neg_manifests --dataset PharDDIE/dataset1
python shared/verify_manifests.py --hash-log EviDDIE/neg_manifests/manifest_hashes.json --manifest-dir EviDDIE/neg_manifests --dataset EviDDIE/dataset1
python shared/audit_leakage.py --dataset PharDDIE/dataset1   # six-part leakage audit
```

### 2. Train PharDDIE (Few-Shot)

```bash
cd PharDDIE

# 5-shot training (dataset1)
python pharddie_trainer.py \
    --dataset dataset1 --few 5 --train_few 5 \
    --batch_size 256 --max_batches 40000 --eval_every 1000 \
    --prefix pharddie_5shot

# 1-shot training
python pharddie_trainer.py \
    --dataset dataset1 --few 1 --train_few 1 \
    --batch_size 256 --max_batches 40000 --eval_every 1000 \
    --prefix pharddie_1shot
```

Per-seed checkpoints are stored under `models/dataset1/models_drugbank_{few}shot_str_seed{seed}/bestmodel`.

### 3. Train EviDDIE (Zero-Shot, Formal Entry)

```bash
cd EviDDIE

# Formal training: BSA GAN + native dual-output EDL comparator (|p_t − z_q|).
# Per-seed independent matcher/generator checkpoints are written to
# models/{prefix}_seed{seed}bestmodel / ...bestmodel_G.
python eviddie_trainer.py --dataset dataset1 --few 10 --train_few 10 \
    --batch_size 256 --max_batches 20000 --seed 19940419 --prefix eviddie_new_s1

# Repeat for seeds 20230801 (s2), 20240115 (s3), 20240520 (s4), 20240910 (s5)
```

### 4. EviDDIE Ablation (Zero-Shot, Frozen Backbone)

```bash
cd EviDDIE

# Frozen-backbone head ablation (per seed): softmax / evi_no_evi / wo_BSA / evi_full.
# Each head is trained from scratch for 5,000 iterations; the production checkpoint
# (jointly trained) is NOT retrained and serves as the horizontal reference.
python eviddie_train_ablation.py --train_seed 19940419 --prefix eviddie_new_s1 \
    --max_iter 5000 --variants softmax,evi_no_evi,wo_BSA,evi_full
# ... repeat for the other four seeds

# Aggregate + significance + figures
python eviddie_ablation_summary.py   # -> results/ablation_summary_eviddie_new.csv
python eviddie_ablation_sigtest.py   # -> results/ablation_sigtest.csv
python eviddie_ablation_figure.py    # -> EviDDIE_Ablation_Study.png (+4-metric supplement)
python eviddie_ablation_curves_figure.py  # -> EviDDIE_Ablation_Curves.png
```

### 5. Export Zero-Shot Predictions (Fixed Manifest, 5 Seeds)

```bash
cd EviDDIE
python eviddie_export_zs_v2.py \
    --variants softmax,evi_no_evi,wo_BSA,full_evi \
    --out_csv predictions_eviddie_new_ablation.csv
# -> results/predictions/predictions_eviddie_new_ablation.csv
#    (per-sample rows with checkpoint_sha256 / eval_manifest_sha256 /
#     event_embedding_sha256 / git_commit; episode manifests saved alongside)
```

### 6. Reproduce Paper Tables & Figures

```bash
# Table 2 — Main few-shot results (1/5-shot)
cd PharDDIE
python pharddie_table2.py

# Table 3 — Zero-shot discrimination + calibration: production EviDDIE + the four
#           frozen-head rows (Softmax / w/o EVI / w/o BSA / frozen EDL head) + TempScale
#           + no-skill; PharDDIE rare rows for comparison
python ../shared/calibration_table.py --csv ../EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv \
    --methods "EviDDIE" "Softmax baseline" "EviDDIE w/o EVI" "EviDDIE w/o BSA" \
    --out ../EviDDIE/results/calibration_table_variants.csv --fig ../EviDDIE/reliability_diagram_new.png

# Table 3 — Frozen EDL head rows (exported from the retrained evi_full heads)
python ../shared/calibration_table.py --csv ../EviDDIE/results/predictions/predictions_evi_full_frozen.csv \
    --methods "EviDDIE (frozen EDL head)" \
    --out ../EviDDIE/results/calibration_table_evi_full.csv

# Table 4 — Selective-referral performance / confidence-based prioritization (unified action semantics, 1-shot;
#           the Random row is computed inside rq3_triage_table.py)
python ../shared/rq3_triage_table.py

# Tables 5/6 — Rebuilt per-signal selective referral (matched coverage / fixed budget)
python ../shared/rq3_selective_referral.py
# -> PharDDIE/results/rq3_rebuilt_PharDDIE.csv (1-shot, per-seed AURC mean±SD, random 95% CI)

# Seed-paired PharDDIE-RareDDIE differences (mean + 95% CI, P1-6)
python ../shared/paired_diff_rareddie.py
# -> PharDDIE/results/paired_diff_PharDDIE_RareDDIE.csv
```

All table scripts abort if the underlying prediction CSVs do not cover the five training seeds, and the export scripts verify checkpoint-hash uniqueness and manifest SHA256 before writing any output.

---

## Evidence Chain

- **Per-sample prediction CSVs** (the sole data sources of the paper's Table 2/3/4/5/6 rows):
  `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (PharDDIE, 5 training seeds)
  and `EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` (EviDDIE current
  architecture, 5 training seeds, fixed evaluation manifest, 4 provenance-hash columns).
- **Manifests**: SHA256-verified negative manifests in `PharDDIE/dataset1/neg_manifests/`
  and `EviDDIE/neg_manifests/` (all five seeds × dev/test/test2).
- **Checkpoint hashes**: `audit/checkpoints_sha256.md` records the SHA256 values of the
  per-seed checkpoints behind the shipped CSVs (binaries not distributed).
- **Training logs**: `audit/training_logs/`.
- **Leakage audits**: six reports in `audit/leakage_reports/` (all hard checks PASS on
  Dataset 1, KG-edge overlap 0).
- **Pipeline**: `reproduce.ps1` runs manifest verification → leakage audit →
  manifest-based exports → table generation, and aborts on any failure (it does not
  train models).

Paper results provenance (table ↔ script ↔ CSV) is documented in [`RESULTS_MAP.md`](RESULTS_MAP.md).
