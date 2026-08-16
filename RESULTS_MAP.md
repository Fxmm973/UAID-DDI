# RESULTS_MAP.md — Paper Results Source Audit Trail

## Version Info
- Last updated: 2026-08-16
- Paper: `fyx_8_16.tex`
- Code: GitHub [`Fxmm973/UAID-DDI`](https://github.com/Fxmm973/UAID-DDI)

---

## Paper-to-Code Quick Reference

| Paper | Script | Type |
|-------|--------|------|
| Datasets summary | — | Text-only table |
| Main results (1/5-shot) | `PharDDIE/pharddie_table2.py` | PharDDIE per-seed rows (shipped CSV) + RareDDIE re-evaluated under the unified protocol + 7 baselines transcribed from published source data |
| Calibration | `PharDDIE/pharddie_table3_complete.py` | PharDDIE per-seed rows + zero-shot rows from the shipped EviDDIE CSV |
| Uncertainty-aware prioritization | `PharDDIE/pharddie_table4_paper.py` | Unified triage semantics, 1-shot settings; AURC / risk-coverage / matched-coverage / fixed-referral budgets |
| Framework schematic | — | `kuangjiatu.jpg` |
| Proxy-channel weight selection | — | `fig_1shot_weight_selection.jpg` / `fig_5shot_weight_selection.jpg` |
| PharDDIE component ablation | — | `1-shot-ablation.jpg` / `5-shot-ablation.jpg` (SHCR / ACI / SRAE removed) |

Note: the zero-shot discrimination table and the post-hoc case study of earlier drafts
have been removed from the paper, and the corresponding legacy scripts have been
removed from the repository (see "Removed content" below).

---

## Main Results (Few-Shot DDI Prediction)

### PharDDIE
- **Source**: `results/predictions/predictions_dataset1_PharDDIE.csv` — per-sample
  predictions of the five independently trained checkpoints
  (`models/dataset1/models_drugbank_{1,5}shot_str_seed{seed}/bestmodel`), evaluated
  with the fixed negative-sampling manifest (eval seed 19940419).
- **Compute script**: `PharDDIE/pharddie_table2.py` (mean ± std across the five
  training seeds; refuses to run unless the CSV covers 5 seeds).

### RareDDIE (re-evaluated under the unified protocol)
- **Source**: the official RareDDIE implementation (github.com/MrPhil/RareDDIE,
  vendored under `PharDDIE/`) was re-trained with five seeds, the same 40k-batch
  budget, dev-AUC checkpoint selection, and evaluated on the same fixed manifest
  (seed 19940419) by `PharDDIE/eval_rareddie_unified.py`.
- **Per-seed results**: `PharDDIE/results/rareddie_seed_{seed}.txt`;
  **aggregation**: `PharDDIE/aggregate_rareddie.py` →
  `PharDDIE/results/rareddie_unified_results.txt` (mean ± population SD, ÷5).
- Its published values (Fig. 3a source data) are shown for reference only.

### Baseline Methods (7 methods)
- **Source**: values transcribed from the published source data of the original
  papers; NOT re-trained or re-evaluated in this study. Methods: META-DDIE,
  GMatching, MRCGNN, MetaR-In, MetaR-Pre, DSN-DDI, KnowDDI.

### Evaluation Protocol
- Event-level splits: train (58 events / 189,287 samples), dev (5 / 2,005),
  test (13 / 408), test2 (10 / 108).
- Negative-sample manifests with SHA256 hashes ensure reproducible evaluation;
  hash records in `PharDDIE/dataset1/neg_manifests/manifest_hashes.json`.
- Six-part leakage audit (`shared/audit_leakage.py`) passes on Dataset 1.
- Ordered triples are kept throughout splitting/negative generation (directional events).

---

## Zero-Shot (EviDDIE)

- **Calibration rows (Table 3)**: computed from
  `EviDDIE/results/predictions/predictions_dataset1_zero_shot_variants.csv` —
  mean ± SD over 3 negative-sampling replicates (seeds 2024, 2025, 2026) on a
  fixed checkpoint. The rows were produced by the earlier implementation of the
  semantic/evidential head (concatenation-based comparator); the public repository
  implements the |p−z| comparator of the paper, and updated numbers under that
  implementation will be deposited together with the released checkpoints.
The paper does not report zero-shot discrimination metrics (the discrimination
table of earlier drafts was removed); the EVI head's measured contribution is
calibration.

---

## Calibration

- **PharDDIE rows**: computed from `results/predictions/predictions_dataset1_PharDDIE.csv`
  (rare-event setting, 1/5-shot); mean ± SD over the five independent training seeds.
- **Zero-shot rows**: see the Zero-Shot section above.
- **Compute script**: `PharDDIE/pharddie_table3_complete.py`.
- **Summary file**: `results/table3new.txt`.

---

## Uncertainty-Aware Prioritization (Triage)

- **Script**: `PharDDIE/pharddie_table4_paper.py`.
- Unified action semantics (automatic = {high-priority, low-priority}, referred =
  {expert referral, deferred review}); coverage = P(u <= tau_u).
- **Signals**: 1-shot uses u_entropy = H(p); 5-shot uses u_latent (SRAE latent
  dispersion). Test scores are mapped through the validation empirical CDF.
- **Source**: `results/predictions/predictions_dataset1_PharDDIE.csv`.
- **Summary file**: `results/table4_paper.txt`.

---

## Architecture ↔ Code Mapping

### PharDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| SHCR | `pharddie_layers.py`:`HiddenChannelReweightingTransformerConv` | 5 fixed hidden-channel indices (0/1/2/46/53) with learnable channel coefficients, 0.7 learned gate : 0.3 channel prior, $1+1.5\gamma$ scaling. A lightweight regularizing prior, not a chemical detector. (`pharm_weight` kept for checkpoint compatibility.) |
| ACI | `pharddie_matcher.py`:`EmbedMatcher.neighbor_encoder()` | Bilinear attention, δ = z_j − z_i, residual gating; structural fallback without KG neighbors |
| SRAE | `pharddie_matcher.py`:`SRAE` (alias `VAE = SRAE`) | η = 1e−2/1e−3, latent dim 64, MSE reconstruction, no KL term |
| Mol. encoder | `pharddie_models.py`:`MVN_DDI` | `initial_node_feature` (Linear) → LayerNorm → ELU → `HiddenChannelReweightingTransformerConv` + SAGPooling |
| Pair scorer | `pharddie_matcher.py`:`EmbedMatcher.forward()` | |z_s − z_q| → MLP → sigmoid |
| Match loss | `pharddie_trainer.py`:`SigmoidLoss` | Sigmoid cross-entropy + SRAE reconstruction (effective weight 0.1) |

### EviDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| Drug encoder | `eviddie_models.py`:`MVN_DDI_Block` | Standard `TransformerConv` (no channel reweighting, no KG aggregation) |
| BSA Generator | `eviddie_matcher.py`:`Generate_Model` | 700→256→512→64, Tanh |
| BSA Critic | `eviddie_matcher.py`:`Distinguish_Model` | 64→512→256→128→1, Sigmoid |
| EVI | `eviddie_matcher.py`:`EmbedMatcher.forward()` | Dual-output head: Softplus → α = e+1, u_EDL = 2/S, EDL loss with annealed KL |
| SRAE | `eviddie_matcher.py`:`SRAE` (alias `VAE = SRAE`) | Same architecture as the PharDDIE SRAE |
| Comparator | `eviddie_matcher.py` | Softplus(|p_t − z_q|) → MLP → R^2_{≥0} |

---

## Evidence Chain (Five-Seed Reproducibility)

1. **Manifests**: `PharDDIE/dataset1/neg_manifests/` + `EviDDIE/neg_manifests/` with
   `manifest_hashes.json`; `shared/verify_manifests.py` verifies SHA256 and per-event
   entry counts.
2. **Per-sample prediction CSVs**: `results/predictions/predictions_dataset1_PharDDIE.csv`
   (PharDDIE, 5 training seeds) and
   `EviDDIE/results/predictions/predictions_dataset1_zero_shot_variants.csv`
   (zero-shot variants, 3 negative-sampling replicates) — the sole data sources of
   the paper's Table 2/3/4 rows and the RQ2 discrimination statements.
3. **Checkpoint hashes**: `audit/checkpoints_sha256.md` records the SHA256 values of
   the per-seed checkpoints behind the shipped CSVs (binaries not distributed).
4. **Training logs**: `audit/training_logs/`.
5. **Leakage audits**: six reports in `audit/leakage_reports/` (all hard checks PASS
   on Dataset 1, KG-edge overlap 0).
6. **Pipeline**: `reproduce.ps1` runs manifest verification → leakage audit →
   manifest-based exports → table generation, and aborts on any failure (it does
   not train models).

---

## Removed Content (earlier drafts)

- **Zero-shot discrimination table and related scripts**: removed from the paper
  and from the repository (the table generator, the direction-audit script, and the
  per-sample CSV-based discrimination reporter were deleted; the paper does not
  report zero-shot discrimination metrics).
- **Post-hoc case study**: removed from the paper (the candidate provenance could
  not be reproduced from the Dataset 2 splits); the case-study scripts were removed
  from the repository.

---

## Large Files Not in Repository

| File | Approx. Size | How to Obtain |
|------|-------------|---------------|
| `DRKG_TransE_entity.npy` | ~30 MB | Regenerate via DRKG training, or contact authors |
| `morgan_dataset*.npz` | ~200 KB | Regenerate via `PharDDIE/fp/save_features.py` |
| Trained checkpoints (`.pth`) | 16–33 MB each | Regenerate with the training scripts, or contact authors; SHA256 in `audit/checkpoints_sha256.md` |
| BioSentVec encoder weights | ~1 GB | Download from the official BioSentVec release (precomputed event embeddings are included) |

---

## Known Limitations

1. **Checkpoint availability**: trained checkpoints are not in the repository due to
   size; they can be regenerated (PharDDIE ≈6 h/seed, EviDDIE ≈3 h/seed on an RTX 4090)
   or obtained from the authors.
2. **Zero-shot tables**: the calibration rows and the RQ2 discrimination statements
   use the shipped fixed-checkpoint CSV (3 negative-sampling replicates, earlier head
   implementation). Regenerating them under the current protocol requires retraining
   the five EviDDIE seeds.
3. **10-shot**: not reported in the paper; no per-seed checkpoints exist.
4. **Drug overlap**: most test-set drugs appear in the training set, and all
   evaluation drugs exist in DRKG. Claims are limited to "unseen-event
   generalization," not "novel-compound generalization" (see `shared/audit_drug_overlap.py`).
5. **DRKG entity embeddings**: not included (file size); relation embeddings are
   provided for symbol initialization.
