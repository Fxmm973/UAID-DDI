# RESULTS_MAP.md — Paper Results Source Audit Trail

## Version Info
- Last updated: 2026-08-07
- Paper: `fyx_ERA_SI_revised_15p_90_lw0804_GPT(1) (2).tex`
- Code: This repository (`PharDDIE_github_quanbu_gaiming_kexingban`)

---

## Table 2 — Main Results (Few-Shot DDI Prediction)

Paper reports AUC / ACC / F1 under 1-shot and 5-shot settings, mean ± SD over five independent training seeds.

### PharDDIE
- **Source**: Local training, five independent random seeds
- **Training script**: `PharDDIE/pharddie_trainer.py`
- **Prediction CSV**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv`
- **Compute script**: `PharDDIE/pharddie_table1.py`
- **Paper values** (Table 2):
  - 1-shot: AUC 0.9675±0.0863, ACC 0.9286±0.0712, F1 0.9271±0.1182
  - 5-shot: AUC 0.9747±0.0158, ACC 0.9310±0.0263, F1 0.9322±0.0172

### RareDDIE (author reproduction)
- **Source**: Local retraining under identical experimental configuration as PharDDIE
- Dataset 1, same event-level splits, DRKG TransE embeddings (128-dim), batch_size=256, lr=0.001, dropout=0.2, Adam optimizer, five independent random seeds
- Architectural difference: RareDDIE uses its original dual-granular structure-adaptive encoder with pair variational representation; PharDDIE uses MME + ACI + SRAE
- Both models share identical data, hyperparameters, and evaluation protocol
- **Paper values** (Table 2):
  - 1-shot: AUC 0.9392±0.0273, ACC 0.8408±0.0202, F1 0.8507±0.0186
  - 5-shot: AUC 0.9679±0.0096, ACC 0.9228±0.0234, F1 0.9270±0.0206

### Remaining 7 Baselines
- **Source**: RareDDIE Nature Communications 2025 Source Data (Excel `41467_2025_59431_MOESM8_ESM.xlsx`, Sheet `fig.3a`)
- **Reference**: Ren et al., "Predicting rare drug-drug interaction events with dual-granular structure-adaptive and pair variational representation", *Nat. Commun.* 16, 3997 (2025)
- Each baseline had 5 independent training seeds in the original paper; mean ± SD computed directly from published source data

| Method | Original Paper | Source Data Location |
|--------|---------------|---------------------|
| META-DDIE | RareDDIE Fig.3a | Sheet fig.3a |
| DSN-DDI | RareDDIE Fig.3a | Sheet fig.3a |
| MRCGNN | RareDDIE Fig.3a | Sheet fig.3a |
| KnowDDI | RareDDIE Fig.3a | Sheet fig.3a |
| GMatching | RareDDIE Fig.3a | Sheet fig.3a |
| MetaR-In | RareDDIE Fig.3a | Sheet fig.3a |
| MetaR-Pre | RareDDIE Fig.3a | Sheet fig.3a |

### Evaluation Protocol
- Event-level data splits (train: 58 common events / 189,287 samples; dev: 5 common events / 2,005 samples; test: 13 fewer events / 408 samples; test2: 10 rare events / 108 samples)
- Pre-generated negative-sample manifests with SHA256 hashes ensure identical negatives across comparisons (seeds: 19940419, 20230801, 20240115, 20240520, 20240910)
- Event-level overlap audit confirms no leakage across train/dev/test/test2

---

## Table 3 — Calibration

Paper reports ECE, Brier score, NLL, and high-confidence error rate. SDs are over five negative-sampling replicates from one fixed checkpoint.

- **PharDDIE**: Computed from `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (per-sample predictions)
- **EviDDIE**: Computed from `EviDDIE/results/predictions/predictions_dataset1_zero_shot_variants.csv`
- **Compute script**: `PharDDIE/pharddie_table2_complete.py`
- **Note**: Results reported as mean ± SD over 5 negative-sampling replicates using one fixed checkpoint (training seed 19940419), NOT 5 independent training runs
- **Paper values** (Table 3):
  - Zero-shot — EviDDIE: ECE 0.1163, Brier 0.2737, NLL 0.7575, high-conf error 0.3229
  - Rare-event — PharDDIE: ECE 0.1343, Brier 0.1454, NLL 0.4864, high-conf error 0.1111
  - Rare-event — PharDDIE + uncertainty: ECE 0.1387, Brier 0.1037, NLL 0.3225, high-conf error 0.0317
- ECE marginal increase on rare-event split (0.1343 → 0.1387) falls within negative-sampling variability

---

## Table 4 — Uncertainty-Aware Agent Prioritization

Paper reports P@10/20/50, referral rate, coverage, and selective risk.

- **Source**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv`
- **Compute script**: `PharDDIE/pharddie_table3_paper.py`
- Uncertainty by VAE latent variance (5/10-shot) or entropy fallback (1-shot)
- Includes matched-coverage selective risk comparison (M8 fix)
- Random ranking: model predictions randomly permuted before applying thresholds
- Triage thresholds: τ_p ∈ [0.70, 0.85], τ_u ∈ [0.30, 0.55] (grid search on validation, ≥30% min coverage)
- **Paper values** (Table 4):
  - Fewer events: probability+uncertainty P@10=1.0000, referral=0.6579, selective risk=0.1817
  - Rare events: probability+uncertainty P@10=0.9333, referral=0.4885, selective risk=0.2151
- Selective-risk reduction partly reflects higher referral rate; presented as illustration of referral–coverage trade-off

---

## Table 5 — Case Study (Internal Consistency vs DrugBank)

- **Source**: EviDDIE inference on Dataset 2 rare-event (test2) split
- **Inference script**: `EviDDIE/eviddie_run_case.py`
- **Table generator**: `EviDDIE/eviddie_case_study.py`
- Cross-split filtering applied; 2 of original top-10 excluded (appeared in data splits)
- DrugBank used exclusively for post hoc annotation; NOT involved in training
- 7 of 8 candidates consistent with DrugBank; 1 candidate (DB03585+DB00288) routed to expert referral

---

## Figure 2 — EviDDIE Ablation Study

- **Data source**: `EviDDIE/results/ablation_curves.csv`
- **Training**: `EviDDIE/eviddie_train_zs_v2.py` (3 variants: softmax / w/o EVI / full EviDDIE)
- **Plotting**: `EviDDIE/eviddie_plot_figure2.py`
- Curves computed on validation events only; held-out fewer/rare events evaluated once after checkpoint selection
- PharDDIE encoder checkpoint used: `EviDDIE/models/dataset1/pharddie_best.pt`

---

## Supplementary Figures

- **S1, S2 (PharDDIE ablation)**: 1-shot and 5-shot component ablation (MME / ACI / SRAE removed)
- **S3 (λ selection)**: Fusion weight sensitivity (`PharDDIE/pharddie_layers.py`, pharmacophore gating ratio 0.7:0.3)

---

## Code Fix Records

| Date | File | Fix |
|------|------|-----|
| 2026-08-04 | `PharDDIE/pharddie_layers.py:147-152` | **M2** — Pharmacophore indices: N→x[:,1], O→x[:,2], C→x[:,0], aromatic→x[:,53], charge→x[:,46]; documented as element-based heuristics |
| 2026-08-04 | `EviDDIE/eviddie_layers.py:151-156` | **M2** — Same pharmacophore fix in EviDDIE copy |
| 2026-08-04 | `PharDDIE/pharddie_trainer.py:241-252` | **M3** — SRAE loss: `loss2` overwrite bug → `loss2_p` + `loss2_n` with symmetric averaging |
| 2026-08-04 | `PharDDIE/pharddie_trainer.py:252` | **M3** — SRAE loss weight applied as `0.5 * (loss2_p + loss2_n)` |
| 2026-08-04 | `EviDDIE/eviddie_trainer.py` | **M4** — Removed proto_loss/var_loss dead code (always zero) |
| 2026-08-04 | `EviDDIE/eviddie_matcher.py` | **M4** — Removed PrototypeAligner import + task_prototypes buffer + post-return dead code |
| 2026-08-04 | `shared/checkpoint.py` (NEW) | **M6** — Safe checkpoint loading: `load_state_dict_safe()` logs missing/unexpected keys, flags critical layer patterns |
| 2026-08-04 | 14 files across PharDDIE/ and EviDDIE/ | **M6** — Replaced `load_state_dict(..., strict=False)` with `load_state_dict_safe()` |
| 2026-08-04 | `shared/neg_manifest.py` (NEW) | **M7** — Fixed negative-sample manifest generation with SHA256 audit; 15 manifests + `manifest_hashes.json` |
| 2026-08-04 | `PharDDIE/pharddie_table3_paper.py` | **M8** — Added matched-coverage selective risk comparison |

### Script Name Mapping (old → current)

| Old Name (pre-rename) | Current File |
|------------------------|--------------|
| `trainer_structure_acc_fp_neigh_VAE_struc.py` | `PharDDIE/pharddie_trainer.py` |
| `compute_table1_final.py` | `PharDDIE/pharddie_table1.py` |
| `compute_table2_complete.py` | `PharDDIE/pharddie_table2_complete.py` |
| `compute_table3_final.py` | `PharDDIE/pharddie_table3_paper.py` |
| `layers.py` (PharDDIE) | `PharDDIE/pharddie_layers.py` |
| `layers.py` (EviDDIE) | `EviDDIE/eviddie_layers.py` |
| `checkpoint_utils.py` | `shared/checkpoint.py` |
| `generate_neg_manifest.py` | `shared/neg_manifest.py` |
| `trainer_...VAE_GAN_struc.py` | `EviDDIE/eviddie_trainer.py` |
| `matcher_...VAE_GAN_struc.py` | `EviDDIE/eviddie_matcher.py` |
| `train_zero_shot_variants_v2.py` | `EviDDIE/eviddie_train_zs_v2.py` |
| `plot_ablation_figure2.py` | `EviDDIE/eviddie_plot_figure2.py` |
| `eval_ablation_fast.py` | `EviDDIE/eviddie_eval_ablation.py` |
| `eviddie_export_dataset1.py` | `EviDDIE/eviddie_export_ds1.py` |
| `run_case_study.py` | `EviDDIE/eviddie_run_case.py` |
| `gen_case_study_table.py` | `EviDDIE/eviddie_case_study.py` |

---

## Architecture ↔ Code Mapping

### PharDDIE
| Paper Module | Code Location | Key Implementation |
|-------------|---------------|-------------------|
| MME | `pharddie_layers.py`:`PharmacophoreAwareTransformerConv` | 5-type pharmacophore gating (0.7:0.3 ratio), node scaling (1+1.5γ) |
| ACI | `pharddie_matcher.py`:`EmbedMatcher.neighbor_encoder()` + `AttentionSelectContext` | Bilinear attention over KG neighbors, differential query, residual gating |
| SRAE | `pharddie_matcher.py`:`VAE` | Asymmetric stochastic bottleneck (η=10⁻²/10⁻³), MSE reconstruction |
| Molecular encoder | `pharddie_models.py`:`MVN_DDI` | `PharmacophoreAwareTransformerConv` + SAGPooling + hierarchical readout |
| Pair scorer | `pharddie_matcher.py`:`EmbedMatcher.forward()` | Latent-code absolute difference → MLP → sigmoid score |
| Matching loss | `pharddie_trainer.py`:`SigmoidLoss` | Sigmoid cross-entropy over positive/negative pairs |

### EviDDIE
| Paper Module | Code Location | Key Implementation |
|-------------|---------------|-------------------|
| BSA (Generator) | `eviddie_matcher.py`: G_φ | 3-layer MLP (768→256→512→64), Tanh, bio-text → prototype |
| BSA (Critic) | `eviddie_matcher.py`: C_ψ | 4-layer MLP (64→512→256→128→1), Sigmoid |
| EVI | `eviddie_matcher.py`: EDL head | Softplus → Dirichlet α = e+1, EDL loss with annealed KL |
| Drug encoder | `eviddie_models.py` | Simplified TransformerConv (no pharmacophore gating, no KG aggregation) |
| Comparator | `eviddie_matcher.py`: MLP_comp | Softplus(|p_t − z_q|) → 3-layer MLP → R²_≥0 |

---

## Key Checkpoints

| Checkpoint | Model | Setting | Location |
|-----------|-------|---------|----------|
| `ph2p0_1shot_40k_FINAL.pth` | PharDDIE | 1-shot, 40k steps | `PharDDIE/models/` |
| `ph2p0_5shot_40k_FINAL.pth` | PharDDIE | 5-shot, 40k steps | `PharDDIE/models/` |
| `ph2p0_10shot_40k_FINAL.pth` | PharDDIE | 10-shot, 40k steps | `PharDDIE/models/` |
| `ph2p0_0shot_40k_FINAL.pth` | EviDDIE | 0-shot, 40k steps | `EviDDIE/models/` |
| `ph2p1_*_40k*/` | PharDDIE | Per-epoch checkpoints with embedded AUC | `PharDDIE/models/` |

---

## Known Limitations

1. **Single-seed checkpoints**: Paper Table 2 uses five independent training seeds (mean ± SD). The published checkpoints represent single-seed runs. Multi-seed retraining from scratch would be needed for exact reproduction of paper SD values.
2. **EviDDIE encoder simplification**: EviDDIE uses a standard TransformerConv graph encoder (no MME pharmacophore gating, no ACI KG aggregation), separate from PharDDIE's full encoder. This is by design (different task requirements), but means the two models do not share a unified encoder.
3. **Training runtime**: Full 40k-step training takes ~8–12 hours on one RTX 4090. The 1-shot verification run (8,000 steps, ~2 hours) in the training log is a lightweight sanity check, not a paper-grade reproduction.
4. **Case study checkpoint sensitivity**: Case study results depend on the specific EviDDIE checkpoint; values may vary with retraining or different random seeds.
5. **Table 3 SD semantics**: Calibration SDs are over negative-sampling replicates (one fixed checkpoint), not independent training runs. This is explicitly noted in the paper.
6. **No LICENSE file**: README claims MIT License but no `LICENSE` file exists in the repository.
