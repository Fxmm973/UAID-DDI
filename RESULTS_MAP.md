# RESULTS_MAP.md — Paper Results Source Audit Trail

## Version Info
- Code: GitHub [`Fxmm973/UAID-DDI`](https://github.com/Fxmm973/UAID-DDI) — all values below
  are reproduced by the scripts listed here, from the per-sample prediction CSVs shipped
  in this repository.

---

## Paper-to-Code Quick Reference

| Paper item | Script | Data source |
|------------|--------|-------------|
| Datasets summary (Table 1) | — | Text-only table |
| Main results 1/5-shot (Table 2) | `PharDDIE/pharddie_table2.py` | `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (+ RareDDIE re-evaluated under the unified protocol; 7 baselines transcribed from published source data) |
| Zero-shot discrimination + calibration (Table 3) | `shared/calibration_table.py` | `EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` → `EviDDIE/results/calibration_table_variants.csv`; PharDDIE rare rows from the PharDDIE CSV above |
| Case study on Dataset 2 (Table 4) | `external/case_study_per_event.py` + `external/case_evidence_upgrade.py` | `external/outputs/predictions_ds2_retrained_0shot.csv` → `external/outputs/case_candidates_dataset2_per_event_v2.csv` (Table 4 = its ten highest-ranked rows by $r$); evidence PMIDs verified in `external/outputs/case_evidence_dataset2_v2.md`; leakage audit `external/audit_case_leakage.py` (PASS 0/25) |
| Reliability diagram (Fig., RQ2) | `shared/calibration_table.py --fig` (equivalently `EviDDIE/eviddie_reliability_figure.py`) | same zero-shot CSV → `EviDDIE/reliability_diagram_new.png` |
| EviDDIE head ablation, final metrics (Fig.) | `EviDDIE/eviddie_ablation_figure.py` | `EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` → `EviDDIE_Ablation_Study.png` (+ `EviDDIE_Ablation_Study_4metrics.png`) |
| EviDDIE head ablation, training dynamics (Fig.) | `EviDDIE/eviddie_ablation_curves_figure.py` | `EviDDIE/results/ablation_curves_eviddie_new_s{1..5}_seed*.csv` + `EviDDIE/results/full_evi_dev_internal.csv` → `EviDDIE_Ablation_Curves.png` |
| Ablation significance tests (text) | `EviDDIE/eviddie_ablation_sigtest.py` | same zero-shot CSV → `EviDDIE/results/ablation_sigtest.csv` |
| Framework schematic | — | `kuangjiatu.jpg` |
| Proxy-channel weight selection (Fig.) | `PharDDIE/pharddie_weight_figure.py` | `PharDDIE/results/validation/weight_sweep.csv` → figure (archived records, pre-unified protocol) |
| PharDDIE component ablation (Fig.) | `PharDDIE/pharddie_ablation_figure.py` | `PharDDIE/results/validation/ablation_results.csv` → figure (archived records, pre-unified protocol) |

---

## Main Results (Few-Shot DDI Prediction) — Table 2

### PharDDIE
- **Source**: `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` — per-samp
  predictions of the five independently trained checkpoints
  (`models/dataset1/models_drugbank_{1,5}shot_str_seed{seed}/bestmodel`), evaluated
  with the fixed negative-sampling manifest (eval seed 19940419).
- **Compute script**: `PharDDIE/pharddie_table2.py` (mean ± population SD, ddof = 0,
  across the five training seeds; refuses to run unless the CSV covers 5 seeds).

### RareDDIE (re-evaluated under the unified protocol)
- **Source**: the official RareDDIE implementation (vendored under `PharDDIE/`) was
  re-trained with five seeds, the same 40k-batch budget, dev-AUC checkpoint selection,
  and evaluated on the same fixed manifest (seed 19940419) by
  `PharDDIE/eval_rareddie_unified.py`.
- **Per-seed results**: `PharDDIE/results/rareddie_seed_{seed}.txt`;
  **aggregation**: `PharDDIE/aggregate_rareddie.py` →
  `PharDDIE/results/rareddie_unified_results.txt` (mean ± population SD).
- **Seed-paired differences (P1-6)**: `shared/paired_diff_rareddie.py` computes per-seed
  PharDDIE−RareDDIE differences on rare events (1-shot/5-shot × AUC/ACC/F1) with
  mean ± 95% CI (t_{0.975,4}) →
  `PharDDIE/results/paired_diff_PharDDIE_RareDDIE.csv`; all intervals include zero.
- Its published values are shown for reference only.

### Baseline Methods (7 methods)
- Values transcribed from the published source data of the original papers; NOT
  re-trained or re-evaluated in this study. Methods: META-DDIE, GMatching, MRCGNN,
  MetaR-In, MetaR-Pre, DSN-DDI, KnowDDI.

### Development-Stage Validation Records (weight sweep & PharDDIE ablation)
- Archived records behind the paper's development-stage figures, parsed into
  `PharDDIE/results/validation/weight_sweep.csv` and
  `PharDDIE/results/validation/ablation_results.csv` from the authors' experiment
  logs (药效团数据.xlsx and the `result_ph2p0_*_40k.txt` records).
- **Protocol caveat**: these runs predate the unified five-seed fixed-manifest
  protocol; their absolute numbers are not comparable with Tables 2/3. The
  full-model reference used when the figures were drawn is not archived
  together with the ablation records. Full details in
  `PharDDIE/results/validation/provenance.md`.

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
- **Table 3 extended columns (P0-5 traceability)**. `shared/calibration_table.py` now
  outputs, for every (setting, method) row, the per-seed detail CSVs
  (`calibration_table_variants_detail.csv`, `calibration_table_evi_full_detail.csv`),
  95% CIs for AUROC/Brier/NLL/ECE (`*_ci95_low/high`, t with 4 df), the linear
  calibration intercept/slope (per-seed regression of the observed positive fraction
  on the predicted probability within 10 equal-width bins; SD uses ddof=0 as in the
  paper), the pooled HCE error numerator/coverage (`hce_err`, `hce_count`,
  `hce_cov_pooled`), and native-vs-TempScale paired t p-values
  (`calibration_table_variants_paired.csv`; the paper's rare-event EviDDIE
  p=0.154/0.060/0.064 are rows of this file).
- **PharDDIE rare-event rows of Table 3**: `PharDDIE/results/validation/
  table3_pharddie_rows.csv`, generated by
  `shared/calibration_table.py --csv PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv
  --methods PharDDIE --settings rare --shot {1,5}`; the HCE coverage denominator is
  the per-shot query sample count (1-shot 37.7%, 5-shot 60.3%).
- **Compute script**: `shared/calibration_table.py` — metrics are computed **per training
  seed first**, aggregated as mean ± SD over the five seeds (P0-5); temperature scaling is
  fitted per seed on that seed's dev (common) rows and applied to its held-out rows;
  includes the analytic no-skill $p{=}0.5$ row (Brier 0.25, NLL ln 2) and prints the
  P0-2 proper-scoring-baseline verdict. Output: `EviDDIE/results/calibration_table_variants.csv`
  (production EviDDIE + the frozen-head rows) and `EviDDIE/results/calibration_table_evi_full.csv`.
- **PharDDIE rare-event rows** (same table, for side-by-side comparison): computed from
  `PharDDIE/results/predictions/predictions_dataset1_PharDDIE.csv` (rare setting,
  1/5-shot), pooled AUROC/AUPRC/ACC + event-macro F1, mean ± SD over the five seeds.
- **Reliability diagram**: `shared/calibration_table.py --fig` →
  `EviDDIE/reliability_diagram_new.png` (Fewer | Rare panels, 10 equal-width predicted-pr
  bins with per-bin sample counts, native evidential bars + per-seed temperature-scaled
  curves). Equivalent standalone generator: `EviDDIE/eviddie_reliability_figure.py`.
- **Frozen-head rows (head-only controlled comparison)**: the four frozen heads
  (Softmax baseline, EviDDIE w/o EVI, EviDDIE w/o BSA, EviDDIE (frozen EDL head))
  share the identical frozen backbone, identical 5,000-iteration budget, identical
  manifest, and identical calibration pipeline, so their differences are attributable
  to the head components alone (P0-2 head-only protocol). Sources:
  `predictions_eviddie_new_ablation.csv` (softmax / w/o EVI / w/o BSA) and
  `EviDDIE/results/predictions/predictions_evi_full_frozen.csv` (frozen EDL head,
  exported from the retrained `evi_full` heads). Compute:
  `shared/calibration_table.py --methods ...` →
  `EviDDIE/results/calibration_table_variants.csv` (with the variant reliability
  figure `EviDDIE/results/reliability_variants.png`) and
  `EviDDIE/results/calibration_table_evi_full.csv`.
- **High-confidence error (HCE)**: classification error among confidence $c=\max(p,1-p)\ge0.9$;
  coverage and counts reported (fewer 6.0\% = 245 of 4080; rare 1.5\% = 16 of 1080, native EviDDIE).
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

## Case Study on Dataset 2 — Table 4 (External Validation)

- **Setup**: Dataset 2 (Lin et al.) shipped as `EviDDIE/dataset2/` — 1,258 drugs,
  80 event types (50 train / 5 dev / 25 held-out), 320,108 records; DRKG `.npy` files
  and the sanitized path graph are copied from Dataset 1 (see
  `external/REPRODUCE_CASE_STUDY.md`). Semantic event overlap with Dataset 1:
  0 exact-text matches; 58/80 events have a BioSentVec-cosine counterpart ≥ 0.7
  (`external/outputs/disjoint_events.json` = the 10 events below that threshold).
- **Training**: `external/train_eviddie_dataset2.py` — same protocol as the Dataset-1
  formal entry (batch 256, lr 1e-3, 20k iterations, dev-AUROC selection), 5 seeds
  **19940419, 20230801, 20240520, 20260201, 20260301**. Protocol seeds
  20240115/20240910 collapse deterministically on Dataset 2 (loss freezes at 1.3333,
  all-positive predictions; reproducible signature) and were replaced under the
  identical protocol — disclosed in the paper; logs in
  `external/outputs/train_logs_ds2/`.
- **Predictions**: `external/eviddie_export_ds2.py --seeds 19940419,20230801,20240520,202
  → `external/outputs/predictions_ds2_retrained_0shot.csv` (18,720 rows = 1,872 test2
  triples × 2 × 5 seeds; per-row checkpoint/manifest/embedding SHA256 + git commit).
  Held-out test2 AUROC **0.5718 ± 0.0125** (mean ± SD over seeds); the cos<0.7
  disjoint-event subset is at chance (0.4984 ± 0.0213) — both reported in the paper.
- **Case selection**: `external/case_study_per_event.py` — per-event top-1 under the
  pre-registered rule $r = p(1-u)$ (five-seed means), excluding the 1,068 Dataset-1
  pair-overlapping test2 pairs → 25 candidates (one per held-out event, all reported)
  in `external/outputs/case_candidates_dataset2_per_event_v2.csv`. **Table 4 = the ten
  highest-ranked rows (by $r$) of this file** (raw five-seed mean probabilities,
  ranking signal).
- **Evidence (three tiers)**: `external/case_evidence_upgrade.py` →
  `external/outputs/case_evidence_dataset2_v2.md`. Direct (pair co-discussed in
  literature): 0; Class-level (single-drug mechanism literature supporting the event
  direction, every PMID verified against its real NCBI title): 15; Not identified: 10.
  Among the Table-4 top-10: 7 class-level / 3 not identified (the paper's "seven …
  supported by class-level mechanistic evidence").
- **Leakage audit**: `external/audit_case_leakage.py` → `external/outputs/case_leakage_audit.json`
  (SHA256-recorded). VERDICT PASS: 0/25 candidates appear in Dataset-2 train/dev tasks
  or in any Dataset-1 task file (pair-level and triple-level).
- **Temperature scaling (supplementary)**: `external/temp_scale_case_table.py`
  (T = 1.364 fitted on the dev split via `external/eviddie_export_ds2_dev.py`) — ranking
  and AUC unchanged; scaled table in
  `external/outputs/case_candidates_dataset2_per_event_v2_ts.csv`.

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
   columns per row) — the data sources of the paper's Tables 2/3. Table 4 (Dataset-2
   case study) is sourced from `external/outputs/predictions_ds2_retrained_0shot.csv`
   (5 training seeds; see the Case Study section).
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
| DrugBank-derived training-task files (`train_tasks.json`) | ~20 MB | License-restrictedth the provided preprocessing scripts |
| Trained checkpoints (`.pth`) | 16–33 MB each | Regenerate with the training scripts, oraudit/checkpoints_sha256.md`; Zenodo deposit on acceptance |
| BioSentVec encoder weights | ~1 GB | Download from the official BioSentVec release (precomputed event embeddings are included) |

---

## Metric Definitions (2026-08-18 audit fixes)

- **ECE**: `M=10` equal-width **predicted-probability** bins over $p$,
  $\mathrm{ECE}=\sum_m |B_m|/N \cdot |\mathrm{acc}(B_m)-\mathrm{conf}(B_m)|$ with
  $\mathrm{acc}(B_m)$ = mean label and $\mathrm{conf}(B_m)$ = mean $p$ per bin
  (paper Eq. ece, revised). Implemented identically in `shared/calibration_table.py`
  and `PharDDIE/pharddie_table3_complete.py`.
- **HCE**: classification error rate $\mathbb{I}(\hat y_i \neq y_i)$ among samples with
  confidence $c=\max(p,1-p)\ge0.9$ (paper Eq. hce). Fixed on 2026-08-18: the previous
  implementation returned the negative-class proportion (`1 - mean(y)`) instead of the
  error rate; the corrected values are in
  `EviDDIE/results/calibration_table_variants.csv` (no-skill row: HCE undefined, `---`).
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
   candidate fraction (fewer 6.0%, rare 1.5%).
