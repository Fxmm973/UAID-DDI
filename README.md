# UAID-DDI: An Uncertainty-Aware Framework for Rare Drug-Drug Interaction Prediction with Rule-Based Triage

Official implementation of the paper *"An Uncertainty-Aware Framework for Rare Drug-Drug Interaction Prediction with Rule-Based Triage"* 

## Overview

UAID-DDI is a two-model framework for predicting rare drug-drug interaction (DDI) events under extreme data scarcity. It outputs both a prediction probability $p$ and an epistemic uncertainty estimate $u$, which drive a downstream **rule-based triage policy** that routes each candidate to one of four actions: High-Priority Review, Expert Referral, Deferred Review, or Low Priority.

| Model | Setting | Key Modules |
|-------|---------|-------------|
| **PharDDIE** | Few-Shot ($K \in \{1,5\}$) | MME (Molecular Motif Extraction) + ACI (Adaptive Context Integration) + SRAE (Stochastic Reconstruction-Regularized Autoencoder) |
| **EviDDIE** | Zero-Shot | BSA (Bio-Semantic Alignment) + EVI (Evidence Variance Inference) |

### PharDDIE — Few-Shot DDI Prediction

- **MME** (Molecular Motif Extraction): `pharddie_layers.py` — Pharmacophore-aware TransformerConv fusing a learned gate $\psi_i$ with pharmacophore signal $\phi_i$ at 0.7:0.3 ratio. Five element-based pharmacophore proxies: H-bond donor (N), acceptor (O), hydrophobic (C), aromatic (RDKit flag), charged (formal charge). Node scaling: $\tilde{h}_i = h_i \odot (1 + 1.5\gamma_i)$.
- **ACI** (Adaptive Context Integration): `pharddie_matcher.py` — Bilinear attention over first-order KG neighbors with differential query $\delta_{ij} = z_i - z_j$, residual-style gating $d_i = \text{ELU}(W_{\text{nei}} c_i + W_{\text{self}} z_i)$.
- **SRAE** (Stochastic Reconstruction-Regularized Autoencoder): `pharddie_matcher.py` — Asymmetric stochastic bottleneck ($\eta = 10^{-2}$ support, $\eta = 10^{-3}$ query). Pair scoring: $\text{MLP}(|z_s - z_q|)$.

### EviDDIE — Zero-Shot DDI Prediction

- **BSA** (Bio-Semantic Alignment): `eviddie_matcher.py` — GAN aligns drug-pair latent codes with BioSentVec event prototypes (700-dim). Generator: $700\to256\to512\to64$ (Tanh). Critic: $64\to512\to256\to128\to1$ (Sigmoid).
- **EVI** (Evidence Variance Inference): `eviddie_matcher.py` — Dirichlet evidential head ($\alpha = e + 1$), EDL loss with annealed KL ($\lambda_t = \min(1, t/10000)$).

---

## Repository Structure

```
UAID-DDI/
├── environment.yml                 # Conda environment specification
├── shared/                         # Shared utilities
│   ├── preprocess.py               # Molecular featurization (RDKit atom/bond features)
│   ├── checkpoint.py               # Safe checkpoint loading with audit logging
│   └── neg_manifest.py             # Negative-sample manifest generation (SHA256 audited)
│
├── PharDDIE/                       # Few-shot model
│   ├── pharddie_args.py            # Hyperparameters & CLI
│   ├── pharddie_dataloader.py      # Episodic data loading & negative sampling
│   ├── pharddie_grapher.py         # Molecular graph construction
│   ├── pharddie_layers.py          # MME: PharmacophoreAwareTransformerConv
│   ├── pharddie_models.py          # MVN_DDI: molecular encoder with SAGPooling
│   ├── pharddie_modules.py         # Support modules (Path, Transformer, attention)
│   ├── pharddie_matcher.py         # EmbedMatcher: ACI + SRAE (VAE) + scorer
│   ├── pharddie_trainer.py         # Training loop with SRAE loss
│   ├── pharddie_train_wo_unc.py    # Ablation: train without uncertainty branch
│   ├── pharddie_tester.py          # Evaluation on test/test2/common_test
│   ├── pharddie_recorder.py        # Experiment result logging
│   ├── pharddie_export.py          # Export predictions to CSV
│   ├── pharddie_table2.py          # Generate Table 2 (main results)
│   ├── pharddie_table3.py          # Generate Table 3 (calibration)
│   ├── pharddie_table3_complete.py # Generate Table 3 (full calibration)
│   ├── pharddie_table4.py          # Generate Table 4 (prioritization)
│   ├── pharddie_table4_paper.py    # Generate Table 4 (paper-format)
│   ├── dataset1/                   # Benchmark dataset (few-shot split)
│   │   ├── train_tasks.json / dev_tasks.json / test_tasks.json / test2_tasks.json
│   │   ├── common_test_tasks.json / uncommon_test_tasks.json
│   │   ├── drug_smiles.csv / dti_entity.csv / dti_rel.csv
│   │   ├── e1rel_e2.json / rel2candidates.json / path_graph
│   │   ├── ent2ids / ent2embids / relation2ids / relation2embids
│   │   ├── test.py
│   │   ├── data/
│   │   └── neg_manifests/          # Pre-generated negative manifests + SHA256 hashes
│   └── dataset2/                   # Case study dataset
│       ├── train_tasks.json / dev_tasks.json / test2_tasks.json
│       ├── drugSMLIES.csv / dti_entity.csv / dti_rel.csv
│       ├── e1rel_e2.json / rel2candidates.json / path_graph
│       ├── ent2ids / ent2embids / relation2ids / relation2embids
│       └── data/
│
└── EviDDIE/                        # Zero-shot model
    ├── eviddie_args.py             # Hyperparameters & CLI
    ├── eviddie_dataloader.py       # Data loading
    ├── eviddie_grapher.py          # Graph construction
    ├── eviddie_layers.py           # MME module copy
    ├── eviddie_models.py           # Molecular encoder (standard TransformerConv)
    ├── eviddie_modules.py          # Support modules
    ├── eviddie_matcher.py          # Matcher with BSA (GAN) + EVI (EDL) heads
    ├── eviddie_trainer.py          # Training loop with GAN + EDL loss
    ├── eviddie_train_zs.py         # Zero-shot training (single variant)
    ├── eviddie_train_ablation.py   # Ablation training (3 variants)
    ├── eviddie_train_variants.py   # Variant training utilities
    ├── eviddie_tester.py           # Evaluation
    ├── eviddie_recorder.py         # Experiment result logging
    ├── eviddie_retriever.py        # Bio-text retrieval & embedding
    ├── eviddie_export_ds1.py       # Export predictions (dataset 1)
    ├── eviddie_export_zs_v2.py     # Export zero-shot predictions (manifest-based)
    ├── eviddie_export_zs_v2.py     # Export zero-shot predictions v2
    ├── eviddie_export_variants.py  # Export ablation variant predictions
    ├── eviddie_eval_ablation.py    # Evaluate ablated checkpoints
    ├── eviddie_plot_figure4.py     # Generate Figure 4 (EviDDIE ablation)
    ├── eviddie_plot_bar.py         # Bar plot utilities
    ├── eviddie_run_case.py         # Case study inference
    ├── eviddie_case_study.py       # Generate Table 5 (case study)
    ├── eviddie_debug.py            # Debugging utilities
    ├── eviddie_verify_ckpt.py      # Checkpoint integrity verification
    ├── train_tasks.json            # EviDDIE training tasks
    ├── neg_manifests/              # Pre-generated negative manifests + SHA256 hashes
    ├── dataset1/                   # Benchmark dataset
    │   ├── dev_tasks.json / test_tasks.json / test2_tasks.json
    │   ├── drug_smiles.csv / dti_entity.csv / dti_rel.csv
    │   ├── e1rel_e2.json / rel2candidates.json / path_graph
    │   ├── ent2ids / ent2embids / relation2ids / relation2embids
    │   └── event_embedding2.json   # BioSentVec semantic embeddings (700-dim)
    └── dataset2/                   # Case study dataset
        ├── dev_tasks.json / test2_tasks.json
        ├── drugSMLIES.csv / dti_entity.csv / dti_rel.csv
        ├── e1rel_e2.json / rel2candidates.json / path_graph
        ├── ent2ids / ent2embids / relation2ids / relation2embids
        └── event_embedding2.json   # BioSentVec semantic embeddings (700-dim)
```

> **Note**: Large files not included in this repository: DRKG TransE entity embeddings (`DRKG_TransE_entity.npy`, ~200 MB), Morgan fingerprint features (`morgan_dataset*.npz`), and trained model checkpoints (`.pth`). The DRKG relation embeddings (`DRKG_TransE_relation.npy`) are provided as they are required for symbol embedding initialization. Checkpoints can be regenerated via the training scripts or obtained from the authors.

---

## System Requirements

Tested on Ubuntu 16.04, CentOS 7, and Windows 11 with Python 3.9 on one NVIDIA RTX 4090 GPU.

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

Negative-sample manifests are pre-generated in `PharDDIE/dataset1/neg_manifests/` and `EviDDIE/neg_manifests/` with SHA256 hashes.

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

### 3. Train EviDDIE (Zero-Shot, Formal Entry)

```bash
cd EviDDIE

# Formal training: GAN (BSA) + evidential head with a native dual-output
# EDL comparator. Per-seed independent matcher/generator checkpoints are
# written to models/{prefix}_seed{seed}bestmodel / ...bestmodel_G.
python eviddie_trainer.py --dataset dataset1 --few 10 --train_few 10 \
    --batch_size 256 --max_batches 20000 --seed 19940419 --prefix eviddie

# Repeat for seeds 20230801, 20240115, 20240520, 20240910
```

### 4. EviDDIE Ablation (Zero-Shot, Frozen Backbone)

```bash
cd EviDDIE

# Frozen-backbone semantic/evidential-head ablation (Figure 4): a
# pre-trained PharDDIE encoder + SRAE are loaded and frozen, and only the
# semantic/evidential head is (re)trained per variant.
python eviddie_train_ablation.py

# Generate Figure 4
python eviddie_plot_figure4.py
```

### 5. Reproduce Paper Tables

```bash
# Table 2 — Main results
cd PharDDIE
python pharddie_table2.py

# Table 3 — Calibration
python pharddie_table3_complete.py

# Table 4 — Uncertainty-aware prioritization
python pharddie_table4_paper.py
```

### 6. Case Study (Table 5)

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

## Paper-to-Code Mapping

| Paper | Script | Description |
|-------|--------|-------------|
| Table 1 | — | Dataset summary (text-only) |
| Table 2 | `PharDDIE/pharddie_table2.py` | Main prediction performance |
| Table 3 | `PharDDIE/pharddie_table3_complete.py` | Calibration metrics |
| Table 4 | `PharDDIE/pharddie_table4_paper.py` | Uncertainty-aware prioritization |
| Table 5 | `EviDDIE/eviddie_case_study.py` | Internal consistency check |
| Figure 1 | — | Framework schematic |
| Figure 2 | — | Fusion weight selection |
| Figure 3 | — | PharDDIE component ablation |
| Figure 4 | `EviDDIE/eviddie_plot_figure4.py` | EviDDIE ablation curves |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Drug embedding dimension | 128 | DRKG TransE pretrained |
| Pharmacophore types | 5 | N (donor), O (acceptor), C (hydrophobic), aromatic, charged |
| Pharmacophore gating | 0.7 (gate) : 0.3 (pharm) | Data-driven over domain prior |
| Node scaling factor | 1.5 | $\tilde{h}_i = h_i \odot (1 + 1.5\gamma_i)$ |
| SRAE $\eta$ (support) | $10^{-2}$ | Stochastic training noise |
| SRAE $\eta$ (query) | $10^{-3}$ | Near-deterministic evaluation |
| SRAE loss weight | 0.2 | Effective weight 0.1 after symmetric averaging |
| Semantic encoder | BioSentVec (700-dim) | PubMed-MIMIC-III, no fine-tuning |
| EDL KL annealing | $\lambda_t = \min(1, t/10000)$ | Gradual prior regularization |
| Batch size | 256 | |
| Optimizer (PharDDIE) | Adam (lr=$10^{-3}$, $\beta_1{=}0.9$, $\beta_2{=}0.999$) | |
| Optimizer (EviDDIE GAN) | Adam (lr=$10^{-4}$) | Generator + Critic |
| Optimizer (EviDDIE main) | Adam (lr=$10^{-4}$, decay=$10^{-5}$) | SRAE + Comparator + EVI |
| Max training steps | 40,000 (PharDDIE), 20,000 (EviDDIE) | |
| Dropout | 0.2 | |
| Negative sampling | 1:1 | Tail-corrupted, event-specific pool |
| KG neighbor limit | 30 per drug | |

---

## Data and Code Availability

- **Datasets**: Derived from DrugBank (version 5.x, license-restricted). Obtain raw records from [DrugBank](https://go.drugbank.com/). Processed splits, identifiers, and preprocessing scripts are provided in this repository.
- **Baseline results**: Transcribed from Ren et al. (Nat. Commun., 2025) published source data (Excel `41467_2025_59431_MOESM8_ESM.xlsx`, Sheet `fig.3a`). See [RESULTS_MAP.md](RESULTS_MAP.md) for the complete audit trail.
- **Semantic embeddings**: Generated using BioSentVec (Chen et al., IEEE ICHI 2019). The 700-dimensional event vectors are in `EviDDIE/dataset*/event_embedding2.json`.
- **Large assets**: DRKG entity embeddings, fingerprint features, and trained checkpoints are not included due to file size. They can be regenerated via the provided scripts or obtained from the authors.

---

