# UAID-DDI: An Uncertainty-Aware Framework for Rare Drug-Drug Interaction Prediction with Rule-Based Triage

Official implementation of the paper *"An Uncertainty-Aware Framework for Rare Drug-Drug Interaction Prediction with Rule-Based Triage"*.

## Overview

UAID-DDI is a two-model framework for predicting rare drug-drug interaction (DDI) events under extreme data scarcity. Each model outputs a prediction probability $p$ together with an uncertainty signal, which drives a downstream **rule-based triage policy**. Three uncertainty signals are distinguished throughout the paper and code:

| Signal | Setting | Meaning |
|--------|---------|---------|
| $u_{\text{entropy}} = H(p)$ | PharDDIE 1-shot | prediction entropy; a confidence-derived baseline (NOT epistemic uncertainty) |
| $u_{\text{latent}}$ | PharDDIE $K\ge 5$ | normalized SRAE **latent dispersion score**; a reconstruction-derived proxy with no KL-based posterior interpretation |
| $u_{\text{EDL}} = 2/S$ | EviDDIE zero-shot | Dirichlet evidential uncertainty (total evidence $S$) |

The triage policy maps each candidate to one of four actions---High-Priority Review, Expert Referral, Deferred Review, Low Priority---under the **unified semantics**: automatic set = {high-priority review, low-priority assignment} ($u \le \tau_u$); referred set = {expert referral, deferred review} ($u > \tau_u$). The paper reports triage results for the 1-shot settings (Table: uncertainty-aware prioritization).

| Model | Setting | Key Modules |
|-------|---------|-------------|
| **PharDDIE** | Few-Shot ($K \in \{1,5\}$) | PPNR (Pharmacophore-Proxy Node Reweighting) + ACI (Adaptive Context Integration) + SRAE (Stochastic Reconstruction-Regularized Autoencoder) |
| **EviDDIE** | Zero-Shot | BSA (Bio-Semantic Alignment) + EVI (Evidence Variance Inference) |

### PharDDIE — Few-Shot DDI Prediction

- **PPNR** (Pharmacophore-Proxy Node Reweighting): `pharddie_layers.py` — `PharmacophoreAwareTransformerConv` fuses a learned data-driven gate $\psi_i$ with a frozen proxy signal $\phi_i$ at a fixed 0.7:0.3 ratio. The proxy reads five fixed channels (0, 1, 2, 46, 53) of the *projected hidden* representation (after `initial_node_feature` linear projection + LayerNorm + ELU); these channels carry no guaranteed chemical meaning and the proxy is best interpreted as a pharmacophore-inspired regularizing prior, **not** a chemical detector. Node scaling: $\tilde{h}_i = h_i \odot (1 + 1.5\gamma_i)$.
- **ACI** (Adaptive Context Integration): `pharddie_matcher.py` — bilinear attention over first-order DRKG neighbors with differential query $\delta_{ij} = z_i - z_j$ and residual-style gating $d_i = \text{ELU}(W_{\text{nei}} c_i + W_{\text{self}} z_i)$; drugs without KG neighbors fall back to the structural branch.
- **SRAE** (Stochastic Reconstruction-Regularized Autoencoder): `pharddie_matcher.py` — asymmetric stochastic bottleneck ($\eta = 10^{-2}$ support, $\eta = 10^{-3}$ query), no KL term and no standard-normal prior; the scale output $\sigma_\phi = \exp(0.5\, l_\phi)$ is a learned noise magnitude, reported as the *latent dispersion score*. Pair scoring: $\text{MLP}(|z_s - z_q|)$.

### EviDDIE — Zero-Shot DDI Prediction

- **BSA** (Bio-Semantic Alignment): `eviddie_matcher.py` — a GAN aligns drug-pair latent codes with BioSentVec event prototypes (700-dim, precomputed in `event_embedding2.json`; encoder weights downloaded from the official BioSentVec release). Generator: $700\to256\to512\to64$ (Tanh). Critic: $64\to512\to256\to128\to1$ (Sigmoid).
- **EVI** (Evidence Variance Inference): `eviddie_matcher.py` — native dual-output Dirichlet evidential head ($\alpha = e + 1$, $u_{\text{EDL}} = 2/S$), EDL loss with annealed KL ($\lambda_t = \min(1, t/10000)$). The formal training entry `eviddie_trainer.py` uses per-seed independent checkpoints (`models/{prefix}_seed{seed}bestmodel{, _G}`); legacy 1-output checkpoints are rejected rather than converted.

---

## Repository Structure

```
UAID-DDI/
├── environment.yml                 # Conda environment (PyTorch 2.0.1+cu118, PyG 2.6.1, RDKit 2025.03.5)
├── reproduce.ps1                   # Fail-fast pipeline: manifest verification -> leakage audit -> exports -> tables (no training)
├── shared/                         # Shared utilities
│   ├── preprocess.py               # Molecular featurization (RDKit atom/bond features)
│   ├── checkpoint.py               # Safe checkpoint loading with audit logging
│   ├── neg_manifest.py             # Negative-sample manifest generation (SHA256 audited)
│   ├── verify_manifests.py         # SHA256 + entry-count verification of all manifests
│   ├── audit_leakage.py            # Six-part leakage audit (support-query / pos-neg / ordered / unordered / cross-split / KG-edge)
│   ├── audit_drug_overlap.py       # Drug-overlap audit across splits
│   └── audit_logger.py             # Audit trail utilities
│
├── PharDDIE/                       # Few-shot model
│   ├── pharddie_args.py            # Hyperparameters & CLI
│   ├── pharddie_dataloader.py      # Episodic data loading
│   ├── pharddie_grapher.py         # Molecular graph construction
│   ├── pharddie_layers.py          # PPNR: PharmacophoreAwareTransformerConv
│   ├── pharddie_models.py          # MVN_DDI: molecular encoder with SAGPooling
│   ├── pharddie_modules.py         # Support modules
│   ├── pharddie_matcher.py         # EmbedMatcher: ACI + SRAE + scorer
│   ├── pharddie_trainer.py         # Training loop with SRAE loss
│   ├── pharddie_train_wo_unc.py    # Ablation: train without uncertainty branch
│   ├── pharddie_tester.py          # Evaluation on test/test2/common_test
│   ├── pharddie_recorder.py        # Experiment result logging
│   ├── pharddie_export.py          # w/o-uncertainty variant export (manifest-based)
│   ├── pharddie_export_full.py     # Main export: fixed manifests, SHA256-verified, SEED-CHAIN-checked
│   ├── pharddie_table2.py          # Table 2 (main results; 8 transcribed baselines)
│   ├── pharddie_table3_complete.py # Calibration table (per-seed aggregation)
│   ├── pharddie_table4_paper.py    # Triage table (unified semantics, 1-shot, AURC/risk-coverage/budget)
│   ├── dataset1/                   # Benchmark dataset (few-shot split) + neg_manifests/ (SHA256-recorded)
│   └── dataset2/                   # Case study dataset
│
├── EviDDIE/                        # Zero-shot model
│   ├── eviddie_args.py             # Hyperparameters & CLI
│   ├── eviddie_dataloader.py       # Data loading
│   ├── eviddie_grapher.py          # Graph construction
│   ├── eviddie_layers.py           # Support layers (no pharmacophore gating, by design)
│   ├── eviddie_models.py           # Molecular encoder (standard TransformerConv)
│   ├── eviddie_modules.py          # Support modules
│   ├── eviddie_matcher.py          # Matcher with BSA (GAN) + EVI (native dual-output EDL)
│   ├── eviddie_trainer.py          # Formal training entry (per-seed checkpoints)
│   ├── eviddie_train_zs.py         # Zero-shot training (single variant)
│   ├── eviddie_train_ablation.py   # Frozen-backbone ablation training (softmax / w/o EVI / full)
│   ├── eviddie_train_variants.py   # Variant training utilities
│   ├── eviddie_tester.py           # Evaluation
│   ├── eviddie_recorder.py         # Experiment result logging
│   ├── eviddie_retriever.py        # Bio-text retrieval & embedding
│   ├── eviddie_export_ds1.py       # Export predictions (dataset 1)
│   ├── eviddie_export_zs_v2.py     # Zero-shot export (manifest-based, per-seed checkpoints)
│   ├── eviddie_export_variants.py  # Export ablation variant predictions
│   ├── eviddie_eval_ablation.py    # Evaluate ablated checkpoints
│   ├── eviddie_table_discrimination.py  # Zero-shot discrimination table (main text)
│   ├── eviddie_plot_figure4.py     # Figure 4 (frozen-backbone ablation curves)
│   ├── eviddie_plot_bar.py         # Bar plot utilities
│   ├── eviddie_run_case.py         # Case study inference
│   ├── eviddie_case_study.py       # Case study table (DrugBank consistency check)
│   ├── eviddie_debug.py            # Debugging utilities
│   ├── eviddie_verify_ckpt.py      # Checkpoint integrity verification
│   ├── train_tasks.json            # EviDDIE training tasks
│   ├── neg_manifests/              # Pre-generated negative manifests + SHA256 hashes
│   ├── dataset1/ dataset2/         # Datasets (incl. event_embedding2.json BioSentVec prototypes)
│   └── results/predictions/        # Fixed-checkpoint zero-shot prediction CSV (paper source)
│
├── results/                        # Table outputs matching the paper
│   ├── table2_final.txt            # Main prediction performance
│   ├── table3_complete_detail.csv  # Calibration, per training seed (PharDDIE rare 1/5-shot)
│   ├── table3new.txt               # Calibration summary (zero-shot + PharDDIE rare rows)
│   └── table4_paper.txt            # Uncertainty-aware prioritization (unified semantics, 1-shot)
│
└── audit/                          # Evidence chain
    ├── checkpoints_sha256.md       # SHA256 manifest of all per-seed checkpoints + BioSentVec embeddings
    ├── environment/environment_info.txt   # Environment lock record
    ├── figure_audit_checklist.md   # Figure-audit checklist
    ├── leakage_reports/            # Six leakage-audit reports (all PASS on Dataset 1)
    └── training_logs/              # Per-seed training logs (PharDDIE 1/5-shot, EviDDIE 0-shot)
```

> **Large files not included**: DRKG TransE entity embeddings (`DRKG_TransE_entity.npy`, ~200 MB), Morgan fingerprint features, and trained model checkpoints (~16-33 MB each). Their SHA256 values are recorded in `audit/checkpoints_sha256.md`; the binaries can be regenerated with the provided scripts or obtained from the authors.

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

Datasets are pre-processed and included in `PharDDIE/dataset1/` and `PharDDIE/dataset2/`. They were derived from DrugBank (version 5.x, license-restricted) and processed following the event-level split protocols from Nyamabo et al. (Briefings in Bioinformatics, 2022) and Lin et al. (Briefings in Bioinformatics, 2021).

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

# Formal training: GAN (BSA) + native dual-output EDL comparator.
# Per-seed independent matcher/generator checkpoints are written to
# models/{prefix}_seed{seed}bestmodel / ...bestmodel_G.
python eviddie_trainer.py --dataset dataset1 --few 10 --train_few 10 \
    --batch_size 256 --max_batches 20000 --seed 19940419 --prefix eviddie

# Repeat for seeds 20230801, 20240115, 20240520, 20240910
```

### 4. EviDDIE Ablation (Zero-Shot, Frozen Backbone)

```bash
cd EviDDIE

# Frozen-backbone semantic/evidential-head ablation (Figure 4): a pre-trained
# encoder + SRAE are loaded and frozen, and only the semantic/evidential head
# is (re)trained per variant (softmax / w/o EVI / full). This ablation isolates
# the head components and does NOT remove the shared molecular encoder or SRAE.
python eviddie_train_ablation.py

# Generate Figure 4
python eviddie_plot_figure4.py
```

### 5. Reproduce Paper Tables

```bash
# Main prediction performance (Table: main results)
cd PharDDIE
python pharddie_table2.py

# Zero-shot discrimination (main-text table) — needs EviDDIE/results/predictions CSV
cd ../EviDDIE
python eviddie_table_discrimination.py

# Calibration (Table: calibration)
cd ../PharDDIE
python pharddie_table3_complete.py

# Uncertainty-aware prioritization (Table: triage, 1-shot, unified semantics)
python pharddie_table4_paper.py
```

All table scripts abort if the underlying prediction CSVs do not cover the five training seeds, and the export scripts verify checkpoint-hash uniqueness and manifest SHA256 before writing any output.

### 6. Case Study (Table: internal consistency check)

```bash
cd EviDDIE
python eviddie_run_case.py
python eviddie_case_study.py
```

### 7. Generate Negative Manifests

```bash
cd shared
python neg_manifest.py --dataset ../PharDDIE/dataset1
```

---


- **Checkpoints**: Not stored in the repository (16-33 MB each). SHA256 values are in `audit/checkpoints_sha256.md`; binaries are available from the corresponding author upon request and will be deposited on Zenodo upon acceptance.
- **Evaluation evidence**: Negative-manifest SHA256 records, per-seed training logs (`audit/training_logs/`), and six leakage-audit reports (`audit/leakage_reports/`) are included.
