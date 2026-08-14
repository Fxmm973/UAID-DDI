# RESULTS_MAP.md — Paper Results Source Audit Trail

## Version Info
- Last updated: 2026-08-14
- Paper: `fyx_8_14.tex`
- Code: GitHub [`Fxmm973/UAID-DDI`](https://github.com/Fxmm973/UAID-DDI)

---

## Paper-to-Code Quick Reference

| Paper | Script | Type |
|-------|--------|------|
| Datasets summary | — | Text-only table |
| Main results (1/5-shot) | `PharDDIE/pharddie_table2.py` | PharDDIE per-seed rows + 8 baselines transcribed from Ren et al. (Nat. Commun., 2025) |
| Zero-shot discrimination | `EviDDIE/eviddie_table_discrimination.py` | Held-out fewer/rare events; source CSV in `EviDDIE/results/predictions/` |
| Calibration | `PharDDIE/pharddie_table3_complete.py` | PharDDIE per-seed rows + zero-shot fixed-checkpoint rows |
| Uncertainty-aware prioritization | `PharDDIE/pharddie_table4_paper.py` | Unified triage semantics, 1-shot settings; AURC / risk-coverage / matched-coverage / fixed-referral budgets |
| Internal consistency check | `EviDDIE/eviddie_case_study.py` | DrugBank post-hoc annotation of held-out candidates |
| Framework schematic | — | `kuangjiatu.jpg` |
| Proxy-weight selection | — | `fig_1shot_weight_selection.jpg` / `fig_5shot_weight_selection.jpg` |
| PharDDIE component ablation | — | `1-shot-ablation.jpg` / `5-shot-ablation.jpg` (PPNR / ACI / SRAE removed) |
| EviDDIE frozen-backbone ablation | `EviDDIE/eviddie_plot_figure4.py` | Validation-event curves (softmax / w/o EVI / full) |

---

## Main Results (Few-Shot DDI Prediction)

### PharDDIE
- **Source**: per-seed prediction CSVs computed from the five independently trained checkpoints (`models/dataset1/models_drugbank_{1,5}shot_str_seed{seed}/bestmodel`), evaluated with the fixed negative-sampling manifest (eval seed 19940419).
- **Training script**: `PharDDIE/pharddie_trainer.py` (five seeds: 19940419, 20230801, 20240115, 20240520, 20240910).
- **Export script**: `PharDDIE/pharddie_export_full.py` (manifest-based; SHA256-verified; SEED-CHAIN-checked).
- **Compute script**: `PharDDIE/pharddie_table2.py` (mean ± std across the five training seeds; refuses to run unless the CSV covers 5 seeds).
- **Checkpoint integrity**: SHA256 values in `audit/checkpoints_sha256.md` (five distinct hashes per shot).

### Baseline Methods (8 methods)
- **Source**: Values transcribed from Ren et al. (Nat. Commun., 2025) published source data (Excel `41467_2025_59431_MOESM8_ESM.xlsx`, Sheet `fig.3a`).
- Methods: RareDDIE, META-DDIE, GMatching, MRCGNN, MetaR-In, MetaR-Pre, DSN-DDI, KnowDDI.
- These values were **not regenerated** in this study; the comparison is an external reference, and no cross-method significance test was performed.

### Evaluation Protocol
- Event-level splits: train (58 events / 189,287 samples), dev (5 / 2,005), test (13 / 408), test2 (10 / 108).
- Negative-sample manifests with SHA256 hashes ensure reproducible evaluation; hash records in `PharDDIE/dataset1/neg_manifests/manifest_hashes.json` (verified 15/15 by `shared/verify_manifests.py`).
- Six-part leakage audit (`shared/audit_leakage.py`) passes on Dataset 1; reports in `audit/leakage_reports/`.
- Ordered triples are kept throughout splitting/negative generation (directional events).

---

## Zero-Shot Discrimination (Main Text)

- **Script**: `EviDDIE/eviddie_table_discrimination.py`.
- **Source CSV**: `EviDDIE/results/predictions/predictions_dataset1_zero_shot_variants.csv` — fixed-checkpoint evaluation over 3 negative-sampling replicates (seeds 2024/2025/2026), the same protocol as the zero-shot rows of the calibration table.
- Zero-shot discrimination on fully held-out event categories is near chance for all variants (expected without molecular support); the full EviDDIE model attains the best rare-event accuracy and macro-F1 and the best fewer-event AUC and macro-F1, while the EVI head's primary measured contribution is calibration (removing it degrades ECE from 0.1163 to 0.2375).
- The variants are frozen-backbone semantic/evidential-head ablations (the shared molecular encoder and SRAE are not removed); see Figure 4.

---

## Calibration

- **PharDDIE rows**: computed from the per-seed prediction CSVs (rare-event setting, 1/5-shot); mean ± SD over the five independent training seeds. Per-seed detail in `results/table3_complete_detail.csv`.
- **Zero-shot rows**: computed from `EviDDIE/results/predictions/predictions_dataset1_zero_shot_variants.csv` — mean ± SD over 3 negative-sampling replicates on a fixed checkpoint.
- **Compute script**: `PharDDIE/pharddie_table3_complete.py` (accepts both `train_seed` and legacy `seed` CSV headers).
- **Summary file**: `results/table3new.txt`.

---

## Uncertainty-Aware Prioritization (Triage)

- **Script**: `PharDDIE/pharddie_table4_paper.py`.
- **Unified action semantics** (matches the paper): automatic = {high-priority review, low-priority assignment} ($u \le \tau_u$); referred = {expert referral, deferred review} ($u > \tau_u$). The same coverage definition is used in the main table and the matched-coverage section.
- **Signals**: 1-shot uses $u_{\text{entropy}} = H(p)$ (confidence-derived baseline); 5-shot uses $u_{\text{latent}}$ (SRAE latent dispersion). Both are mapped to percentile ranks in [0,1]; test scores are mapped through the validation empirical CDF so $\tau_u$ transfers across sets (robust to heavy-tailed scores).
- **Reported in the paper**: 1-shot settings only (per author decision).
- **Script outputs**: main table (Random rows include referral/coverage/selective risk under the same mask), per-setting $\tau_p$/$\tau_u$, risk-coverage curves + AURC, matched-coverage comparison, fixed-referral-budget comparison, exact automatic/referred counts.
- **Source**: `results/predictions/predictions_dataset1_PharDDIE.csv` (regenerated by `pharddie_export_full.py`).

---

## Internal Consistency Check (Case Study)

- **Source**: EviDDIE inference on Dataset 2 test2 held-out positive instances.
- **Scripts**: `EviDDIE/eviddie_run_case.py` (inference), `EviDDIE/eviddie_case_study.py` (table).
- Candidates are the held-out rare-split positives themselves; candidates appearing in the training/validation/test splits are excluded from the cross-split filter (test2 is the target candidate pool by design).
- DrugBank (version 5.x) is used exclusively for post-hoc annotation, not training; 7 of 8 retained candidates consistent with DrugBank.
- This is a retrospective descriptive analysis, not independent external validation.

---

## Figure 4 — EviDDIE Frozen-Backbone Ablation

- **Training**: `EviDDIE/eviddie_train_ablation.py` (variants: softmax / w/o EVI / full EviDDIE; frozen backbone: a pre-trained encoder + SRAE are loaded and frozen, only the semantic/evidential head is trained per variant).
- **Plotting**: `EviDDIE/eviddie_plot_figure4.py`.
- Curves from validation events only; held-out events are evaluated after checkpoint selection.
- EviDDIE uses standard `TransformerConv` (no PPNR gating, no ACI KG aggregation), by design for the zero-shot setting.
- Semantic encoder: BioSentVec (700-dim); embeddings in `EviDDIE/dataset*/event_embedding2.json` (SHA256 in `audit/checkpoints_sha256.md`).

---

## Architecture ↔ Code Mapping

### PharDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| PPNR | `pharddie_layers.py`:`PharmacophoreAwareTransformerConv` | 5 fixed hidden-channel proxy scores (0/1/2/46/53 of the projected representation), 0.7 learned gate : 0.3 proxy prior, $1+1.5\gamma$ scaling. A regularizing prior, not a chemical detector. |
| ACI | `pharddie_matcher.py`:`EmbedMatcher.neighbor_encoder()` | Bilinear attention, $\delta=z_i-z_j$, residual gating; structural fallback without KG neighbors |
| SRAE | `pharddie_matcher.py`:`VAE` | $\eta=10^{-2}/10^{-3}$, latent dim 64, MSE reconstruction, **no KL term**; scale output = latent dispersion score |
| Mol. encoder | `pharddie_models.py`:`MVN_DDI` | `initial_node_feature` (Linear) → LayerNorm → ELU → `PharmacophoreAwareTransformerConv` + SAGPooling |
| Pair scorer | `pharddie_matcher.py`:`EmbedMatcher.forward()` | $|z_s-z_q| \to$ MLP $\to$ sigmoid |
| Match loss | `pharddie_trainer.py`:`SigmoidLoss` | Sigmoid cross-entropy + SRAE reconstruction (effective weight 0.1) |

### EviDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| Drug encoder | `eviddie_models.py`:`MVN_DDI_Block` | Standard `TransformerConv` (no PPNR, no KG aggregation) |
| BSA Generator | `eviddie_matcher.py`:`Generate_Model` | $700\to256\to512\to64$, Tanh |
| BSA Critic | `eviddie_matcher.py`:`Distinguish_Model` | $64\to512\to256\to128\to1$, Sigmoid |
| EVI | `eviddie_matcher.py`:`EmbedMatcher.forward()` | Native dual-output head: Softplus $\to \alpha=e+1$, $u_{\text{EDL}}=2/S$, EDL loss with annealed KL |
| SRAE | `eviddie_matcher.py`:`VAE` | Same architecture as PharDDIE SRAE |
| Comparator | `eviddie_matcher.py` | Softplus($|p_t - z_q|$) $\to$ MLP $\to \mathbb{R}^2_{\ge 0}$ |

---

## Evidence Chain (Five-Seed Reproducibility)

1. **Manifests**: `PharDDIE/dataset1/neg_manifests/` + `EviDDIE/neg_manifests/` with `manifest_hashes.json`; `shared/verify_manifests.py` verifies SHA256 and per-event entry counts (30/30 files pass).
2. **Checkpoints**: per-seed SHA256 in `audit/checkpoints_sha256.md` (PharDDIE 1/5-shot: 5 distinct hashes each; EviDDIE 0-shot: 5 main + 5 generator).
3. **Training logs**: `audit/training_logs/` — PharDDIE 1-shot 5/5, 5-shot 4/5 (seed 19940419 log not retained; its checkpoint hash is on record), EviDDIE 0-shot 5/5.
4. **SEED-CHAIN checks**: the export scripts verify 5 training seeds → 5 distinct checkpoint paths → 5 distinct checkpoint hashes → identical eval-manifest hash, and abort otherwise; the table scripts refuse to aggregate CSVs that do not cover 5 training seeds.
5. **Leakage audits**: six reports in `audit/leakage_reports/` (support–query, positive–negative, ordered-triple, unordered-pair, cross-split, KG-edge; all hard checks PASS on Dataset 1, KG-edge overlap 0).
6. **Pipeline**: `reproduce.ps1` runs manifest verification → leakage audit → manifest-based exports → table generation, and **aborts immediately on any failure** (it does not train models).

---

## Large Files Not in Repository

| File | Approx. Size | How to Obtain |
|------|-------------|---------------|
| `DRKG_TransE_entity.npy` | ~200 MB | Regenerate via DRKG training, or contact authors |
| `morgan_dataset*.npz` | ~50 MB each | Regenerate via `PharDDIE/fp/save_features.py` |
| Trained checkpoints (`.pth`) | 16–33 MB each | Regenerate with the training scripts, or contact authors; SHA256 in `audit/checkpoints_sha256.md` |
| BioSentVec encoder weights | ~1 GB | Download from the official BioSentVec release (precomputed event embeddings are included) |

---

## Known Limitations

1. **Checkpoint availability**: trained checkpoints are not in the repository due to size; they can be regenerated (PharDDIE ≈6 h/seed, EviDDIE ≈3 h/seed on an RTX 4090) or obtained from the authors.
2. **10-shot**: not reported in the paper; only one per-seed checkpoint (seed 19940419) exists, so the strict export does not cover 10-shot.
3. **Zero-shot tables**: the calibration and discrimination rows use the fixed-checkpoint predictions shipped in `EviDDIE/results/predictions/` (3 negative-sampling replicates). Regenerating them under the current P0-7 protocol (native dual-output EDL head, no inference noise) requires retraining the five EviDDIE seeds.
4. **Main-table reproducibility note**: the printed Table 2/3 PharDDIE values were produced under the evaluation protocol in effect at the time of writing; the current manifest-based export reproduces the evaluation exactly for the fixed manifest (seed 19940419), but regenerated aggregates can differ from the printed values if any protocol detail (e.g., negative-manifest content) changed between runs. Both the printed values and the scripts are provided for transparency.
5. **Drug overlap**: most test-set drugs appear in the training set, and all evaluation drugs exist in DRKG. Claims are limited to "unseen-event generalization," not "novel-compound generalization" (see `shared/audit_drug_overlap.py`).
6. **DRKG entity embeddings**: not included (file size); relation embeddings are provided for symbol initialization.
7. **Case study**: retrospective analysis of held-out positives, not independent external validation. DrugBank evidence verified at the event level (drug pair + event type).
