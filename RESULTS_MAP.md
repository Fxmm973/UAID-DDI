# RESULTS_MAP.md — Paper Results Source Audit Trail

## Version Info
- Last updated: 2026-08-07
- Paper: `fyx_ERA_SI_revised_15p_90_lw0804_GPT(1) (2).tex` → `fyx_8_7.tex`
- Code: GitHub [`Fxmm973/UAID-DDI`](https://github.com/Fxmm973/UAID-DDI)

---

## Table 2 — Main Results (Few-Shot DDI Prediction)

Paper reports AUC / ACC / F1 under 1-shot and 5-shot settings, mean ± SD over five independent training seeds.

### PharDDIE
- **Source**: Local training, five independent random seeds (19940419, 20230801, 20240115, 20240520, 20240910)
- **Training script**: `PharDDIE/pharddie_trainer.py`
- **Prediction CSV**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (generated locally)
- **Compute script**: `PharDDIE/pharddie_table1.py`
- **Paper values** (Table 2):
  - 1-shot: AUC 0.9675±0.0863, ACC 0.9286±0.0712, F1 0.9271±0.1182
  - 5-shot: AUC 0.9747±0.0158, ACC 0.9310±0.0263, F1 0.9322±0.0172

### RareDDIE (author reproduction)
- **Source**: Local retraining under identical experimental configuration as PharDDIE
- Dataset 1, same event-level splits, DRKG TransE embeddings (128-dim), batch_size=256, lr=0.001, dropout=0.2, Adam optimizer, five independent random seeds
- Architectural difference: RareDDIE uses its original dual-granular structure-adaptive encoder with pair variational representation; PharDDIE uses MME + ACI + SRAE
- **Paper values** (Table 2):
  - 1-shot: AUC 0.9392±0.0273, ACC 0.8408±0.0202, F1 0.8507±0.0186
  - 5-shot: AUC 0.9679±0.0096, ACC 0.9228±0.0234, F1 0.9270±0.0206

### Remaining 7 Baselines
- **Source**: RareDDIE Nature Communications 2025 Source Data (Excel `41467_2025_59431_MOESM8_ESM.xlsx`, Sheet `fig.3a`)
- **Reference**: Ren et al., *Nat. Commun.* 16, 3997 (2025)
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

---

## Table 3 — Calibration

- **PharDDIE**: Computed from `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (per-sample predictions, generated locally)
- **EviDDIE**: Computed from `EviDDIE/results/predictions/predictions_dataset1_zero_shot_variants.csv` (generated locally)
- **Compute script**: `PharDDIE/pharddie_table2_complete.py`
- **Note**: Results reported as mean ± SD over 5 negative-sampling replicates using one fixed checkpoint (training seed 19940419), NOT 5 independent training runs
- **Paper values** (Table 3):
  - Zero-shot — EviDDIE: ECE 0.1163, Brier 0.2737, NLL 0.7575, high-conf error 0.3229
  - Rare-event — PharDDIE: ECE 0.1343, Brier 0.1454, NLL 0.4864, high-conf error 0.1111
  - Rare-event — PharDDIE + uncertainty: ECE 0.1387, Brier 0.1037, NLL 0.3225, high-conf error 0.0317

---

## Table 4 — Uncertainty-Aware Agent Prioritization

- **Source**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (generated locally)
- **Compute script**: `PharDDIE/pharddie_table3_paper.py`
- Uncertainty by VAE latent variance (5/10-shot) or entropy fallback (1-shot)
- Triage thresholds: τ_p ∈ [0.70, 0.85], τ_u ∈ [0.30, 0.55] (grid search on validation, ≥30% min coverage)
- Includes matched-coverage selective risk comparison (M8 fix)
- **Paper values** (Table 4):
  - Fewer events: probability+uncertainty P@10=1.0000, referral=0.6579, selective risk=0.1817
  - Rare events: probability+uncertainty P@10=0.9333, referral=0.4885, selective risk=0.2151

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

- **Data source**: `EviDDIE/results/ablation_curves.csv` (generated locally)
- **Training**: `EviDDIE/eviddie_train_zs_v2.py` (3 variants: softmax / w/o EVI / full EviDDIE)
- **Plotting**: `EviDDIE/eviddie_plot_figure2.py`
- Curves computed on validation events only; held-out fewer/rare events evaluated once after checkpoint selection
- PharDDIE encoder checkpoint used: `EviDDIE/models/dataset1/pharddie_best.pt` (generated locally)

---

## Supplementary Figures

- **S1, S2 (PharDDIE ablation)**: 1-shot and 5-shot component ablation (MME / ACI / SRAE removed)
- **S3 (λ selection)**: Fusion weight sensitivity (`PharDDIE/pharddie_layers.py`, pharmacophore gating ratio 0.7:0.3)

---

## Code Fix Records

| Date | File | Fix |
|------|------|-----|
| 2026-08-04 | `PharDDIE/pharddie_layers.py` | **M2** — Pharmacophore indices: N→x[:,1], O→x[:,2], C→x[:,0], aromatic→x[:,53], charge→x[:,46] |
| 2026-08-04 | `EviDDIE/eviddie_layers.py` | **M2** — Same pharmacophore fix in EviDDIE copy |
| 2026-08-04 | `PharDDIE/pharddie_trainer.py` | **M3** — SRAE loss: `loss2` overwrite bug → `loss2_p` + `loss2_n` with symmetric averaging |
| 2026-08-04 | `EviDDIE/eviddie_trainer.py` | **M4** — Removed proto_loss/var_loss dead code (always zero) |
| 2026-08-04 | `EviDDIE/eviddie_matcher.py` | **M4** — Removed PrototypeAligner import + task_prototypes buffer + post-return dead code |
| 2026-08-04 | `shared/checkpoint.py` | **M6** — Safe checkpoint loading: `load_state_dict_safe()` logs missing/unexpected keys |
| 2026-08-04 | `shared/neg_manifest.py` | **M7** — Fixed negative-sample manifest generation with SHA256 audit |
| 2026-08-04 | `PharDDIE/pharddie_table3_paper.py` | **M8** — Added matched-coverage selective risk comparison |

---

## Architecture ↔ Code Mapping

### PharDDIE
| Paper Module | Code Location | Key Implementation |
|-------------|---------------|-------------------|
| MME | `PharDDIE/pharddie_layers.py`:`PharmacophoreAwareTransformerConv` | 5-type pharmacophore gating (0.7:0.3 ratio), node scaling (1+1.5γ) |
| ACI | `PharDDIE/pharddie_matcher.py`:`EmbedMatcher.neighbor_encoder()` | Bilinear attention over KG neighbors, differential query δ=z_i−z_j, residual gating |
| SRAE | `PharDDIE/pharddie_matcher.py`:`VAE` | Asymmetric stochastic bottleneck (η=10⁻²/10⁻³, latent dim=64), MSE reconstruction |
| Molecular encoder | `PharDDIE/pharddie_models.py`:`MVN_DDI` | `PharmacophoreAwareTransformerConv` + SAGPooling + hierarchical readout |
| Pair scorer | `PharDDIE/pharddie_matcher.py`:`EmbedMatcher.forward()` | Latent-code absolute difference → MLP → sigmoid score |
| Matching loss | `PharDDIE/pharddie_trainer.py`:`SigmoidLoss` | Sigmoid cross-entropy over positive/negative pairs |

### EviDDIE
| Paper Module | Code Location | Key Implementation |
|-------------|---------------|-------------------|
| Drug encoder | `EviDDIE/eviddie_models.py`:`MVN_DDI_Block` | Standard `TransformerConv` (no pharmacophore gating, no KG aggregation) |
| BSA (Generator) | `EviDDIE/eviddie_matcher.py`:`Generate_Model` | 3-layer MLP (768→256→512→64), Tanh |
| BSA (Critic) | `EviDDIE/eviddie_matcher.py`:`Distinguish_Model` | 4-layer MLP (64→512→256→128→1), Sigmoid |
| EVI | `EviDDIE/eviddie_matcher.py`:`EmbedMatcher.forward()` | Softplus → Dirichlet α=e+1, EDL loss with annealed KL (λ_t=min(1,t/10000)) |
| SRAE | `EviDDIE/eviddie_matcher.py`:`VAE` | Same architecture as PharDDIE SRAE |
| Comparator | `EviDDIE/eviddie_matcher.py`:`EmbedMatcher.forward()` | Softplus(|p_t − z_q|) → MLP → R²_≥0 |

---

## Training Seed Documentation

Table 2 main results use five independent random seeds. To reproduce the full 5-seed experiment:

```bash
cd PharDDIE
for seed in 19940419 20230801 20240115 20240520 20240910; do
    python pharddie_trainer.py \
        --dataset dataset1 --few 5 --train_few 5 \
        --batch_size 256 --max_batches 40000 --eval_every 1000 \
        --prefix pharddie_5shot_seed${seed} --seed ${seed}
done
python pharddie_table1.py  # aggregates results across all 5 seeds
```

Negative-sample manifests for all five seeds are pre-generated in `PharDDIE/dataset1/neg_manifests/`.

---

## Known Limitations

1. **Checkpoint availability**: Trained model checkpoints are not included in this repository due to file size. The training scripts, random seeds, and hyperparameters documented above enable full reproduction through retraining (~8–12 hours on one RTX 4090).
2. **Single-seed published checkpoints**: Paper Table 2 uses five independent training seeds (mean ± SD). The prediction CSVs and table outputs reflect the aggregated 5-seed results; individual per-seed checkpoints were not all retained.
3. **EviDDIE encoder**: EviDDIE uses a standard `TransformerConv` graph encoder (no MME pharmacophore gating, no ACI KG aggregation), separate from PharDDIE's full encoder path. This is by design for the zero-shot setting and is documented in the paper.
4. **Table 3 SD semantics**: Calibration standard deviations are over negative-sampling replicates (one fixed checkpoint, seed 19940419), not independent training runs. This is explicitly noted in the Table 3 footnote in the paper.
5. **Case study checkpoint sensitivity**: Case study results (Table 5) depend on the specific EviDDIE checkpoint; values may vary with retraining or different random seeds.
6. **DRKG embeddings**: The `DRKG_TransE_entity.npy` and `DRKG_TransE_relation.npy` files (~200 MB each) are not included in the GitHub repository due to size limits. They can be regenerated using the DRKG training scripts or obtained from the authors.
