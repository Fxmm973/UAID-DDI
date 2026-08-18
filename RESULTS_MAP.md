# RESULTS_MAP.md — Paper Results Source Audit Trail

## Version Info
- Last updated: 2026-08-18
- Paper: `fyx_8_16(3) (2).tex` (reviewer-revision with P0-1 … P0-5 fixes)
- Code: GitHub [`Fxmm973/UAID-DDI`](https://github.com/Fxmm973/UAID-DDI) — all values below
  are reproduced by the scripts listed here, from the per-sample prediction CSVs shipped
  in this repository.

---

## Paper-to-Code Quick Reference

| Paper item | Script | Data source |
|------------|--------|-------------|
| Datasets summary (Table 1) | — | Text-only table |
| Main results 1/5-shot (Table 2) | `PharDDIE/pharddie_table2.py` | `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (+ RareDDIE re-evaluated under the unified protocol; 7 baselines transcribed from published source data) |
| Zero-shot discrimination + calibration (Table 3) | `shared/calibration_table.py` | `EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` → `EviDDIE/results/calibration_table_new.csv`; PharDDIE rare rows from the PharDDIE CSV above |
| Uncertainty-aware prioritization (Table 4) | `shared/rq3_triage_table.py` + `shared/rq3_triage_priority_random.py` | `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (1-shot) |
| Selective referral, matched coverage (Table 5) | `shared/rq3_selective_referral.py` | `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (1-shot) → `PharDDIE/results/rq3_rebuilt_PharDDIE.csv` |
| Selective referral, fixed budgets (Table 6) | `shared/rq3_selective_referral.py` | same CSV / same output file |
| Reliability diagram (Fig., RQ2) | `shared/calibration_table.py --fig` (equivalently `EviDDIE/eviddie_reliability_figure.py`) | same zero-shot CSV → `EviDDIE/reliability_diagram_new.png` |
| EviDDIE head ablation, final metrics (Fig.) | `EviDDIE/eviddie_ablation_figure.py` | `EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` → `EviDDIE_Ablation_Study.png` (+ `EviDDIE_Ablation_Study_4metrics.png`) |
| EviDDIE head ablation, training dynamics (Fig.) | `EviDDIE/eviddie_ablation_curves_figure.py` | `EviDDIE/results/ablation_curves_eviddie_new_s{1..5}_seed*.csv` + `EviDDIE/results/full_evi_dev_internal.csv` → `EviDDIE_Ablation_Curves.png` |
| Ablation significance tests (text) | `EviDDIE/eviddie_ablation_sigtest.py` | same zero-shot CSV → `EviDDIE/results/ablation_sigtest.csv` |
| Framework schematic | — | `kuangjiatu.jpg` |
| Proxy-channel weight selection | — | `fig_1shot_weight_selection.jpg` / `fig_5shot_weight_selection.jpg` |
| PharDDIE component ablation | — | `1-shot-ablation.jpg` / `5-shot-ablation.jpg` (SHCR / ACI / SRAE removed) |

---

## Main Results (Few-Shot DDI Prediction) — Table 2

### PharDDIE
- **Source**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` — per-sample
  predictions of the five independently trained checkpoints
  (`models/dataset1/models_drugbank_{1,5}shot_str_seed{seed}/bestmodel`), evaluated
  with the fixed negative-sampling manifest (eval seed 19940419).
- **Compute script**: `PharDDIE/pharddie_table2.py` (mean ± SD across the five training
  seeds; refuses to run unless the CSV covers 5 seeds).

### RareDDIE (re-evaluated under the unified protocol)
- **Source**: the official RareDDIE implementation (vendored under `PharDDIE/`) was
  re-trained with five seeds, the same 40k-batch budget, dev-AUC checkpoint selection,
  and evaluated on the same fixed manifest (seed 19940419) by
  `PharDDIE/eval_rareddie_unified.py`.
- **Per-seed results**: `PharDDIE/results/rareddie_seed_{seed}.txt`;
  **aggregation**: `PharDDIE/aggregate_rareddie.py` →
  `PharDDIE/results/rareddie_unified_results.txt` (mean ± population SD).
- Its published values are shown for reference only.

### Baseline Methods (7 methods)
- Values transcribed from the published source data of the original papers; NOT
  re-trained or re-evaluated in this study. Methods: META-DDIE, GMatching, MRCGNN,
  MetaR-In, MetaR-Pre, DSN-DDI, KnowDDI.

### Evaluation Protocol
- Event-level splits: train (58 events / 189,287 samples), dev (5 / 2,005),
  test (13 / 408), test2 (10 / 108).
- Negative-sample manifests with SHA256 hashes ensure reproducible evaluation;
  hash records in `PharDDIE/dataset1/neg_manifests/manifest_hashes.json`.
- Six-part leakage audit (`shared/audit_leakage.py`) passes on Dataset 1.
- Ordered triples are kept throughout splitting/negative generation (directional events).

---

## Zero-Shot Discrimination & Calibration (EviDDIE) — Table 3, Reliability Fig.

Current architecture: **|p_t − z_q| absolute-difference comparator + native dual-output
EDL head**, five independent training seeds (19940419, 20230801, 20240115, 20240520,
20240910; prefixes `eviddie_new_s1 … s5`), one fixed evaluation manifest (seed 19940419).
Inference uses the raw BioSentVec prototypes (no semantic noise). This supersedes the
earlier concatenation-based comparator used by pre-revision drafts.

- **Source**: `EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` —
  per-sample predictions of all five seeds on dev/test/test2 (method rows `EviDDIE`
  native, `EviDDIE w/o EVI`, `EviDDIE w/o BSA`, `Softmax baseline`); every row carries
  `checkpoint_sha256`, `eval_manifest_sha256`, `event_embedding_sha256`, and `git_commit`.
  Export entry point: `EviDDIE/eviddie_export_zs_v2.py`.
- **Compute script**: `shared/calibration_table.py` — metrics are computed **per training
  seed first**, aggregated as mean ± SD over the five seeds (P0-5); temperature scaling is
  fitted per seed on that seed's dev (common) rows and applied to its held-out rows;
  includes the analytic no-skill $p{=}0.5$ row (Brier 0.25, NLL ln 2) and prints the
  P0-2 proper-scoring-baseline verdict. Output: `EviDDIE/results/calibration_table_new.csv`.
- **PharDDIE rare-event rows** (same table, for side-by-side comparison): computed from
  `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (rare setting,
  1/5-shot), pooled AUROC/AUPRC/ACC + event-macro F1, mean ± SD over the five seeds.
- **Reliability diagram**: `shared/calibration_table.py --fig` →
  `EviDDIE/reliability_diagram_new.png` (Fewer | Rare panels, 10 equal-width confidence
  bins with per-bin sample counts, native evidential bars + per-seed temperature-scaled
  curves). Equivalent standalone generator: `EviDDIE/eviddie_reliability_figure.py`.
- **High-confidence error (HCE)**: confidence > 0.9; coverage and counts reported
  (fewer 5.9% = 240 of 4080; rare 1.7% = 18 of 1080, native EviDDIE).
- **Evaluation episode manifests** (P0-5): `EviDDIE/results/predictions/episode_manifests/`
  (one JSON per training seed).

---

## Zero-Shot Frozen-Backbone Ablation — Ablation Figures & Significance

- **Training**: `EviDDIE/eviddie_train_ablation.py` — all backbone parameters frozen
  (KG neighbor encoder, MVN_DDI, SRAE, GAN generator $G_\phi$); four comparator heads
  trained from scratch for 5,000 iterations per variant: `softmax` (cross-entropy),
  `evi_no_evi` (MSE without the EVI KL regularizer), `wo_BSA` (trainable linear prototype
  projection replacing $G_\phi$), `evi_full` (complete EDL loss with annealed KL).
  Per-seed training curves: `EviDDIE/results/ablation_curves_eviddie_new_s{1..5}_seed*.csv`;
  per-seed heads: `models/ablation_eviddie_new_s{1..5}_seed*/fc_*.pt` (binaries not
  distributed; SHA256 in `audit/checkpoints_sha256.md`).
- **Production reference line**: `EviDDIE/eviddie_eval_full_dev.py` evaluates the
  untouched production checkpoint (jointly trained head + backbone) under the identical
  internal dev protocol → `EviDDIE/results/full_evi_dev_internal.csv`.
- **Final metrics figure**: `EviDDIE/eviddie_ablation_figure.py` →
  `EviDDIE_Ablation_Study.png` (AUROC + F1 across common/fewer/rare, 4 variants,
  mean ± SD, paired-$t$ stars) and `EviDDIE_Ablation_Study_4metrics.png` (supplement).
- **Training-dynamics figure**: `EviDDIE/eviddie_ablation_curves_figure.py` →
  `EviDDIE_Ablation_Curves.png`.
- **Significance**: `EviDDIE/eviddie_ablation_sigtest.py` → `EviDDIE/results/ablation_sigtest.csv`
  (paired $t$-tests per setting × metric: w/o BSA significant on common and fewer F1,
  w/o EVI significant on rare F1, $p<0.05$; AUROC/AUPRC differences $p>0.3$).
- **Summary**: `EviDDIE/eviddie_ablation_summary.py` → `EviDDIE/results/ablation_summary_eviddie_new.csv`.

---

## Uncertainty-Aware Prioritization (Triage) — Table 4

- **Scripts**: `shared/rq3_triage_table.py` (main rows) and
  `shared/rq3_triage_priority_random.py` (independent Random row: per seed × coverage,
  200 random referral repetitions, mean ± 95% CI).
- **Source**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (1-shot).
- Unified action semantics (automatic = {high-priority, low-priority}, referred =
  {expert referral, deferred review}); coverage = P(u ≤ τ_u); referral rate = P(u > τ_u).
- **Signals**: 1-shot uses $u_{\text{entropy}} = H(p)$ (a confidence-derived baseline,
  not epistemic uncertainty); 5-shot uses $u_{\text{latent}}$ (SRAE latent dispersion).
  Test scores are mapped through the validation empirical CDF.
- **Summary file**: `PharDDIE/results/table4_paper.txt` (legacy generator
  `PharDDIE/pharddie_table4_paper.py` kept for provenance).

---

## Selective Referral, Rebuilt Per-Signal — Tables 5 & 6

- **Script**: `shared/rq3_selective_referral.py` (P0-3 protocol, 2026-08-16).
  - Signals are separate strategies, each keeping/referring candidates by its own
    ranking: raw positive score $p$ (candidate-priority semantics), MSP = max(p, 1−p),
    margin |p−0.5|, entropy H(p), the model's native $u$ (latent dispersion for
    PharDDIE), and **true random referral**.
  - MSP / margin / entropy rank-equivalence for binary classification is verified
    numerically (Spearman $\rho = \pm 1$) and printed.
  - Risk–coverage curves and AURC are computed **per seed first**, then aggregated as
    mean ± SD (never pooled) — per P0-3.
  - Error-detection AUROC/AUPRC is reported for the native $u$ signal and for MSP.
  - Random baseline: independent referral sets per (seed, coverage/budget), 200
    repetitions, mean ± 95% CI.
- **Source**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (1-shot).
- **Output**: `PharDDIE/results/rq3_rebuilt_PharDDIE.csv` — the sole data source of the
  paper's Tables 5 and 6 (e.g., fewer p-AURC 0.2006 ± 0.0046, MSP-AURC 0.4127 ± 0.0478;
  30%-coverage risks 0.0574 / 0.4236 / 0.4000 / random 0.5010 ± 0.0038). Budgets of
  10%/30%/50% correspond to automatic coverages of 90%/70%/50%.

---

## Zero-Shot RQ3 Exploration (not in the paper)

`EviDDIE/results/rq3_eviddie_new.csv` and `EviDDIE/results/rq3_rebuilt_eviddie_zs.csv`
contain the same per-signal selective-referral analysis applied to the **zero-shot
EviDDIE** predictions (shot = 0). The paper's Tables 5/6 use the 1-shot PharDDIE
analysis above; these zero-shot files are exploratory and are kept for reference only.

---

## Architecture ↔ Code Mapping

### PharDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| SHCR | `pharddie_layers.py`:`HiddenChannelReweightingTransformerConv` | 5 fixed hidden-channel indices (0/1/2/46/53) with learnable channel coefficients, 0.7 learned gate : 0.3 channel prior, $1+1.5\gamma$ scaling. A lightweight regularizing prior, not a chemical detector. |
| ACI | `pharddie_matcher.py`:`EmbedMatcher.neighbor_encoder()` | Bilinear attention, δ = z_j − z_i, residual gating; structural fallback without KG neighbors |
| SRAE | `pharddie_matcher.py`:`SRAE` | η = 1e−2/1e−3, latent dim 64, MSE reconstruction, no KL term |
| Mol. encoder | `pharddie_models.py`:`MVN_DDI` | `initial_node_feature` (Linear) → LayerNorm → ELU → `HiddenChannelReweightingTransformerConv` + SAGPooling |
| Pair scorer | `pharddie_matcher.py`:`EmbedMatcher.forward()` | \|z_s − z_q\| → MLP → sigmoid |
| Match loss | `pharddie_trainer.py` | Sigmoid cross-entropy + SRAE reconstruction |

### EviDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| Drug encoder | `eviddie_models.py` | Standard `TransformerConv` (no channel reweighting, no KG aggregation) |
| BSA Generator | `eviddie_matcher.py`:`Generate_Model` | 700→256→512→64, Tanh |
| BSA Critic | `eviddie_matcher.py`:`Distinguish_Model` | 64→512→256→128→1, Sigmoid |
| EVI | `eviddie_matcher.py` | Dual-output evidential head: Softplus → α = e+1, u_EDL = 2/S, EDL loss with annealed KL |
| Comparator | `eviddie_matcher.py` | Softplus(MLP(\|p_t − z_q\|)) → R^2_{≥0}; legacy single-output/concat checkpoints are rejected, never converted |
| Frozen-backbone ablation heads | `eviddie_train_ablation.py` | softmax / evi_no_evi / wo_BSA / evi_full trained on frozen backbones; production checkpoint = horizontal reference |

---

## Evidence Chain (Five-Seed Reproducibility)

1. **Manifests**: `PharDDIE/dataset1/neg_manifests/` + `EviDDIE/neg_manifests/` with
   `manifest_hashes.json`; `shared/verify_manifests.py` verifies SHA256 and per-event
   entry counts.
2. **Per-sample prediction CSVs**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv`
   (PharDDIE, 5 training seeds) and
   `EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv`
   (EviDDIE current architecture, 5 training seeds, fixed manifest, 4 provenance-hash
   columns per row) — the sole data sources of the paper's Tables 2/3/4/5/6.
3. **Checkpoint hashes**: `audit/checkpoints_sha256.md` records the SHA256 values of
   the per-seed checkpoints behind the shipped CSVs (binaries not distributed).
4. **Training logs**: `audit/training_logs/`.
5. **Leakage audits**: six reports in `audit/leakage_reports/` (all hard checks PASS
   on Dataset 1, KG-edge overlap 0).
6. **Pipeline**: `reproduce.ps1` runs manifest verification → leakage audit →
   manifest-based exports → table generation, and aborts on any failure (it does
   not train models).

---

## Large Files Not in Repository

| File | Approx. Size | How to Obtain |
|------|-------------|---------------|
| `DRKG_TransE_entity.npy` | ~200 MB | Third-party pre-trained artifact from the DRKG release; relation embeddings are included |
| DrugBank-derived training-task files (`train_tasks.json`) | ~20 MB | License-restricted; regenerate from DrugBank with the provided preprocessing scripts |
| Trained checkpoints (`.pth`) | 16–33 MB each | Regenerate with the training scripts, or contact authors; SHA256 in `audit/checkpoints_sha256.md`; Zenodo deposit on acceptance |
| BioSentVec encoder weights | ~1 GB | Download from the official BioSentVec release (precomputed event embeddings are included) |

---

## Known Limitations

1. **Checkpoint availability**: trained checkpoints are not in the repository due to
   size; they can be regenerated (PharDDIE ≈6 h/seed, EviDDIE ≈3 h/seed on an RTX 4090)
   or obtained from the authors.
2. **10-shot**: not reported in the paper; no per-seed checkpoints exist.
3. **Drug overlap**: most test-set drugs appear in the training set, and all evaluation
   drugs exist in DRKG. Claims are limited to "unseen-event generalization," not
   "novel-compound generalization" (see `shared/audit_drug_overlap.py`).
4. **Ablation scope**: the frozen-backbone ablation isolates the head components only;
   it does not remove the shared molecular encoder or SRAE modules.
5. **Zero-shot calibration claim**: per the P0-2 revision, the paper's claim is limited
   to "reduces overconfidence relative to the softmax baseline together with usable
   discrimination"; high-confidence error remains elevated but affects only a small
   candidate fraction (fewer 5.9%, rare 1.7%).
