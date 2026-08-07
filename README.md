# UAID-DDI: An Uncertainty-Aware Framework for Rare Drug-Drug Interaction Prediction with Rule-Based Triage

Official implementation of the paper *"An Uncertainty-Aware Framework for Rare Drug-Drug Interaction Prediction with Rule-Based Triage"* (Electronic Research Archive, 2026).

## Overview

UAID-DDI is a two-model framework for predicting rare drug-drug interaction (DDI) events under extreme data scarcity. It outputs both a prediction probability $p$ and an epistemic uncertainty estimate $u$, which drive a downstream **rule-based triage policy** that routes each candidate to one of four actions: High-Priority Review, Expert Referral, Deferred Review, or Low Priority.

| Model | Setting | Key Modules |
|-------|---------|-------------|
| **PharDDIE** | Few-Shot ($K \in \{1,5\}$) | MME (Molecular Motif Extraction) + ACI (Adaptive Context Integration) + SRAE (Stochastic Reconstruction-Regularized Autoencoder) |
| **EviDDIE** | Zero-Shot | BSA (Bio-Semantic Alignment) + EVI (Evidence Variance Inference) |

### PharDDIE — Few-Shot DDI Prediction

- **MME** (Molecular Motif Extraction): `pharddie_layers.py` — Pharmacophore-aware TransformerConv that fuses a learned data-driven gate $\psi_i$ with a domain-informed pharmacophore signal $\phi_i$ at a 0.7:0.3 ratio. Five pharmacophore types are proxied by element-level heuristics: H-bond donor (N), H-bond acceptor (O), hydrophobic (C), aromatic (RDKit flag), and charged (formal charge). Node representations are adaptively scaled as $\tilde{h}_i = h_i \odot (1 + 1.5\gamma_i)$.
- **ACI** (Adaptive Context Integration): `pharddie_matcher.py` — Bilinear attention over first-order knowledge-graph neighbors (genes, proteins, diseases) with a differential query $\delta_{ij} = z_i - z_j$, producing dual-granular drug representations via residual-style gating $d_i = \text{ELU}(W_{\text{nei}} c_i + W_{\text{self}} z_i)$.
- **SRAE** (Stochastic Reconstruction-Regularized Autoencoder): `pharddie_matcher.py` — Asymmetric stochastic bottleneck ($\eta = 10^{-2}$ for support, $\eta_{\text{eval}} = 10^{-3}$ for query) that learns a compact latent space; reconstruction loss acts as a regularizer on the pair representation. Pair scoring uses latent-code absolute difference passed through an MLP: $\text{score} = \text{MLP}(|z_s - z_q|)$.

### EviDDIE — Zero-Shot DDI Prediction

- **BSA** (Bio-Semantic Alignment): `eviddie_matcher.py` — Adversarial GAN aligns drug-pair latent codes with clinical event prototypes derived from a PubMed/MIMIC-III pretrained sentence encoder (768-dim), enabling generalization to unseen event types.
- **EVI** (Evidence Variance Inference): `eviddie_matcher.py` — Evidential deep learning head (Dirichlet) replaces softmax, producing calibrated probabilities and epistemic uncertainty from total evidence $S = \alpha_0 + \alpha_1$.

---

## Repository Structure

```
PharDDIE_github/
├── README.md
├── RESULTS_MAP.md                  # Audit trail: results source & code fixes
├── environment.yml                 # Conda environment specification
│
├── shared/                         # Shared utilities
│   ├── preprocess.py               # Molecular featurization (RDKit atom/bond features)
│   ├── checkpoint.py               # Safe checkpoint loading with audit logging (M6 fix)
│   └── neg_manifest.py             # Fixed negative-sample manifest generation (M7 fix)
│
├── PharDDIE/                       # Few-shot model
│   ├── pharddie_args.py            # Hyperparameters & CLI
│   ├── pharddie_dataloader.py      # Episodic data loading & negative sampling
│   ├── pharddie_grapher.py         # Molecular graph construction
│   ├── pharddie_layers.py          # MME: PharmacophoreAwareTransformerConv (M2 fix site)
│   ├── pharddie_models.py          # MVN_DDI: molecular encoder with SAGPooling readout
│   ├── pharddie_modules.py         # Support modules (Path encoder, Transformer, attention)
│   ├── pharddie_matcher.py         # EmbedMatcher: ACI neighbor aggregator + SRAE (VAE) + scorer
│   ├── pharddie_trainer.py         # Training loop with SRAE loss (M3 fix site)
│   ├── pharddie_train_wo_unc.py    # Ablation: train without uncertainty branch
│   ├── pharddie_tester.py          # Evaluation on test/test2/common_test sets
│   ├── pharddie_recorder.py        # Experiment result logging
│   ├── pharddie_export.py          # Export predictions to CSV
│   ├── pharddie_table1.py          # Compute main results table
│   ├── pharddie_table2.py          # Compute calibration table
│   ├── pharddie_table2_complete.py # Compute full calibration results
│   ├── pharddie_table3.py          # Compute prioritization table
│   ├── pharddie_table3_paper.py    # Compute paper-format prioritization (M8 fix)
│   ├── fp/
│   │   ├── save_features.py        # Morgan fingerprint extraction
│   │   └── features/               # Precomputed fingerprint files
│   ├── dataset1/                   # Benchmark dataset (few-shot split)
│   │   ├── train_tasks.json / dev_tasks.json / test_tasks.json / test2_tasks.json
│   │   ├── drug_smiles.csv / path_graph / e1rel_e2.json / rel2candidates.json
│   │   ├── DRKG_TransE_entity.npy / DRKG_TransE_relation.npy
│   │   ├── ent2ids / ent2embids / relation2ids / relation2embids
│   │   └── neg_manifests/          # Pre-generated negative manifests + SHA256 hash log
│   ├── dataset2/                   # Dataset for case study
│   ├── models/                     # Trained checkpoints (ph2p0_*, ph2p1_*)
│   └── results/                    # Prediction CSVs & computed table outputs
│
└── EviDDIE/                        # Zero-shot model
    ├── eviddie_args.py             # Hyperparameters & CLI
    ├── eviddie_dataloader.py       # Data loading
    ├── eviddie_grapher.py          # Graph construction
    ├── eviddie_layers.py           # MME module copy (M2 fix site)
    ├── eviddie_models.py           # Molecular encoder
    ├── eviddie_modules.py          # Support modules
    ├── eviddie_matcher.py          # Matcher with BSA (GAN) + EVI (EDL) heads (M4 fix site)
    ├── eviddie_trainer.py          # Training loop with GAN + EDL loss (M4 fix site)
    ├── eviddie_train_zs.py         # Zero-shot training (single variant)
    ├── eviddie_train_zs_v2.py      # Ablation training (3 variants: softmax / w/o EVI / full)
    ├── eviddie_train_variants.py   # Variant training utilities
    ├── eviddie_tester.py           # Evaluation
    ├── eviddie_recorder.py         # Experiment result logging
    ├── eviddie_retriever.py        # Bio-text retrieval & embedding
    ├── eviddie_export_ds1.py       # Export predictions (dataset 1)
    ├── eviddie_export_zs.py        # Export zero-shot predictions
    ├── eviddie_export_zs_v2.py     # Export zero-shot predictions v2
    ├── eviddie_export_variants.py  # Export ablation variant predictions
    ├── eviddie_eval_ablation.py    # Evaluate ablated checkpoints
    ├── eviddie_plot_figure2.py     # Generate Figure 2 (ablation curves)
    ├── eviddie_plot_bar.py         # Bar plot utilities
    ├── eviddie_run_case.py         # Case study inference
    ├── eviddie_case_study.py       # Generate case study table
    ├── eviddie_debug.py            # Debugging utilities
    ├── eviddie_verify_ckpt.py      # Checkpoint integrity verification
    ├── fp/features/                # Precomputed fingerprints
    ├── dataset1/                   # Benchmark dataset (zero-shot split)
    ├── dataset2/                   # Case study dataset
    ├── models/                     # Trained checkpoints
    └── results/                    # Prediction CSVs & ablation curves
```

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric rdkit-pypi numpy pandas scikit-learn tqdm tensorboardX
```

---

## Quick Start

### 1. Data Preparation

Datasets are pre-processed and included in `PharDDIE/dataset1/` and `EviDDIE/dataset1/`. They were derived from DrugBank (version 5.x, license-restricted) and processed following the few-shot/zero-shot event-level split protocols from Nyamabo et al. (Briefings in Bioinformatics, 2022) and Lin et al. (Briefings in Bioinformatics, 2021).

Negative-sample manifests are pre-generated in `dataset1/neg_manifests/` with SHA256 hashes recorded in `manifest_hashes.json`.

### 2. Train PharDDIE (Few-Shot)

```bash
cd PharDDIE

# 5-shot training (dataset1)
python pharddie_trainer.py \
    --dataset dataset1 --few 5 --train_few 5 \
    --batch_size 256 --max_batches 40000 --eval_every 1000 \
    --prefix pharddie_5shot --seed 19940419

# 1-shot training
python pharddie_trainer.py \
    --dataset dataset1 --few 1 --train_few 1 \
    --batch_size 256 --max_batches 40000 --eval_every 1000 \
    --prefix pharddie_1shot --seed 19940419
```

### 3. EviDDIE Ablation (Zero-Shot)

```bash
cd EviDDIE

# Copy PharDDIE encoder checkpoint
mkdir -p models/dataset1
cp ../PharDDIE/models/ph2p0_5shot_40kbestmodel models/dataset1/pharddie_best.pt

# Run ablation training (3 variants: softmax / w/o EVI / full)
python eviddie_train_zs_v2.py

# Generate Figure 2
python eviddie_plot_figure2.py
```

### 4. Reproduce Paper Tables

```bash
# Table 2 — Main results (few-shot DDI prediction)
cd PharDDIE
python pharddie_table1.py

# Table 3 — Calibration
python pharddie_table2_complete.py

# Table 4 — Uncertainty-aware agent prioritization
python pharddie_table3_paper.py
```

### 5. Case Study

```bash
cd EviDDIE
python eviddie_run_case.py
python eviddie_case_study.py
```

### 6. Generate Negative Manifests

```bash
cd shared
python neg_manifest.py --dataset ../PharDDIE/dataset1
python neg_manifest.py --dataset ../EviDDIE/dataset1
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Drug embedding dimension | 128 | DRKG TransE pretrained |
| Pharmacophore types | 5 | H-bond donor (N), acceptor (O), hydrophobic (C), aromatic, charged |
| Pharmacophore gating ratio | 0.7 (gate) : 0.3 (pharm) | Data-driven $\psi_i$ dominates over domain prior $\phi_i$ |
| Node scaling factor | 1.5 | $\tilde{h}_i = h_i \odot (1 + 1.5\gamma_i)$ |
| SRAE perturbation $\eta$ (support) | $10^{-2}$ | Stochastic training noise |
| SRAE perturbation $\eta$ (query) | $10^{-3}$ | Near-deterministic evaluation |
| SRAE loss weight $\lambda_{\text{SRAE}}$ | 0.2 | Reconstruction regularizer |
| EDL KL annealing | $\lambda_t = \min(1, t/10000)$ | Gradual prior regularization |
| Batch size | 256 (PharDDIE), 256 (EviDDIE) | |
| Optimizer (PharDDIE) | Adam (lr=$10^{-3}$, $\beta_1{=}0.9$, $\beta_2{=}0.999$) | |
| Optimizer (EviDDIE GAN) | Adam (lr=$10^{-4}$) | Generator + Critic |
| Optimizer (EviDDIE main) | Adam (lr=$10^{-4}$, weight decay=$10^{-5}$) | SRAE + Comparator + EVI head |
| Max training steps | 40,000 (PharDDIE), 20,000 (EviDDIE) | |
| Dropout | 0.2 | |
| Negative sampling | 1:1 (positive:negative) | Tail-corrupted, event-specific candidate pool |
| KG neighbor limit | 30 per drug | |
| LR scheduler | MultiStepLR (milestone=200000, $\gamma=0.5$) | |
| Evaluation interval | Every 1,000 batches | Validation AUC for checkpoint selection |

---

## Key Code Changes from Preprint

This repository incorporates fixes identified during peer review:

| Fix | File | Description |
|-----|------|-------------|
| M2 | `pharddie_layers.py` / `eviddie_layers.py` | Pharmacophore indices corrected: N→x[:,1], O→x[:,2], C→x[:,0], aromatic→x[:,53], charge→x[:,46]; documented as element-based heuristics |
| M3 | `pharddie_trainer.py` | SRAE loss: separate `loss2_p` and `loss2_n`, symmetric averaging $\frac{1}{2}(\mathcal{L}_{\text{SRAE}}^{\text{pos}} + \mathcal{L}_{\text{SRAE}}^{\text{neg}})$ |
| M4 | `eviddie_matcher.py` / `eviddie_trainer.py` | Removed EMA prototype alignment dead code (proto_loss $\equiv$ 0) |
| M6 | `shared/checkpoint.py` | Safe checkpoint loading with auditable missing/unexpected key logging |
| M7 | `shared/neg_manifest.py` | Fixed negative-sample manifest generation with SHA256 audit |
| M8 | `pharddie_table3_paper.py` | Added matched-coverage selective risk comparison |

---

## Data and Code Availability

- **Datasets**: Derived from DrugBank (version 5.x, license-restricted). Raw DrugBank records must be obtained from [DrugBank](https://go.drugbank.com/). Processed split indices, DrugBank identifiers, preprocessing scripts, model configurations, random seeds, and table-reproduction scripts are provided in this repository.
- **Baseline results**: Sourced from RareDDIE Nature Communications 2025 Source Data (Excel `41467_2025_59431_MOESM8_ESM.xlsx`, Sheet `fig.3a`). Refer to [RESULTS_MAP.md](RESULTS_MAP.md) for the complete audit trail.
- **Precomputed artifacts**: Prediction CSVs, trained checkpoints, and table outputs are included under `PharDDIE/results/` and `PharDDIE/models/`, enabling table reproduction without retraining.
- **Data provenance**: See [RESULTS_MAP.md](RESULTS_MAP.md) for per-table data sources, script mappings, and fix records.

---

## Citation

```bibtex
@article{li2026uaiddi,
  title={An Uncertainty-Aware Framework for Rare Drug-Drug Interaction Prediction with Rule-Based Triage},
  author={Li, Wen and Fu, Yanxiu and Long, Quan and Zhang, Wei and Li, Zejun},
  journal={Electronic Research Archive},
  year={2026}
}
```

## License

MIT License.

---

## References

1. Nyamabo AK, Yu H, Liu Z, Shi JY. Drug–drug interaction prediction with learnable size-adaptive molecular substructures. *Briefings in Bioinformatics*, 2022.
2. Lin S, et al. MDF-SA-DDI: predicting drug–drug interaction events based on multi-source drug fusion. *Briefings in Bioinformatics*, 2021.
3. Ren Z, et al. Predicting rare drug-drug interaction events with dual-granular structure-adaptive and pair variational representation. *Nature Communications*, 2025.
4. Wishart DS, et al. DrugBank 5.0: a major update to the DrugBank database for 2018. *Nucleic Acids Research*, 2018.
