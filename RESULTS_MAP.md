# RESULTS_MAP.md — Paper Results Source Audit Trail

## Version Info
- Last updated: 2026-08-08
- Paper: `fyx_8_8.tex`
- Code: GitHub [`Fxmm973/UAID-DDI`](https://github.com/Fxmm973/UAID-DDI)

---

## Paper-to-Code Quick Reference

| Paper | Script | Type |
|-------|--------|------|
| Table 1 | — | Dataset summary (text-only table) |
| Table 2 | `PharDDIE/pharddie_table2.py` | Main prediction performance |
| Table 3 | `PharDDIE/pharddie_table3_complete.py` | Calibration metrics |
| Table 4 | `PharDDIE/pharddie_table4_paper.py` | Uncertainty-aware prioritization |
| Table 5 | `EviDDIE/eviddie_case_study.py` | Internal consistency check |
| Figure 1 | — | Framework schematic |
| Figure 2 | — | Fusion weight selection |
| Figure 3 | — | PharDDIE component ablation |
| Figure 4 | `EviDDIE/eviddie_plot_figure4.py` | EviDDIE ablation curves |

---

## Table 2 — Main Results (Few-Shot DDI Prediction)

### PharDDIE
- **Source**: Training under the specified configuration (Dataset 1 rare-event split, DRKG TransE 128-dim, batch_size=256, lr=0.001, dropout=0.2, Adam, 40k iterations, seed 19940419)
- **Training script**: `PharDDIE/pharddie_trainer.py`
- **Compute script**: `PharDDIE/pharddie_table2.py`
- **Paper values** (rare-event test split):
  - 1-shot: AUC 0.9675, ACC 0.9286, F1 0.9271
  - 5-shot: AUC 0.9747, ACC 0.9310, F1 0.9322

### Baseline Methods
- **Source**: Values transcribed from Ren et al. (Nat. Commun., 2025) published source data
- **Data file**: `41467_2025_59431_MOESM8_ESM.xlsx`, Sheet `fig.3a`
- Methods: RareDDIE, META-DDIE, GMatching, MetaR-In, MetaR-Pre, DSN-DDI, MRCGNN, KnowDDI
- These values were not regenerated in this study; the comparison is an external reference

### Evaluation Protocol
- Event-level splits: train (58 events / 189,287 samples), dev (5 / 2,005), test (13 / 408), test2 (10 / 108)
- Negative-sample manifests with SHA256 hashes ensure reproducible evaluation
- Five manifest seeds: 19940419, 20230801, 20240115, 20240520, 20240910
- Event-level overlap audit confirms no leakage across splits

---

## Table 3 — Calibration

- **PharDDIE**: Computed from prediction CSVs (rare-event, single fixed checkpoint, seed 19940419)
- **EviDDIE**: Computed from zero-shot prediction CSVs
- **Compute script**: `PharDDIE/pharddie_table3_complete.py`
- **Important**: Results are mean ± SD over 5 negative-sampling replicates using one fixed checkpoint. They do NOT represent 5 independent training runs. This distinction is noted in the paper.
- **Paper values**:
  - Zero-shot EviDDIE: ECE 0.1163, Brier 0.2737, NLL 0.7575, high-conf error 0.3229
  - Rare-event PharDDIE: ECE 0.1343, Brier 0.1454, NLL 0.4864, high-conf error 0.1111
  - Rare-event PharDDIE + uncertainty: ECE 0.1387, Brier 0.1037, NLL 0.3225, high-conf error 0.0317

---

## Table 4 — Uncertainty-Aware Prioritization

- **Source**: PharDDIE prediction CSVs
- **Compute script**: `PharDDIE/pharddie_table4_paper.py`
- Uncertainty by VAE latent variance (5/10-shot) or entropy fallback (1-shot)
- Triage thresholds: $\tau_p \in [0.70, 0.85]$, $\tau_u \in [0.30, 0.55]$, selected on validation
- The paper presents these as an illustration of the referral-coverage trade-off
- **Paper values**:
  - Fewer: prob+unc P@10=1.0000, referral=0.6579, selective risk=0.1817
  - Rare: prob+unc P@10=0.9333, referral=0.4885, selective risk=0.2151

---

## Table 5 — Internal Consistency Check vs DrugBank

- **Source**: EviDDIE inference on Dataset 2 test2 held-out positive instances
- **Scripts**: `EviDDIE/eviddie_run_case.py` (inference), `EviDDIE/eviddie_case_study.py` (table)
- DrugBank (version 5.x) used exclusively for post hoc annotation, not training
- This is a retrospective descriptive analysis, not independent external validation
- 7 of 8 retained candidates consistent with DrugBank

---

## Figure 4 — EviDDIE Ablation

- **Training**: `EviDDIE/eviddie_train_ablation.py` (3 variants: softmax / w/o EVI / full EviDDIE)
- **Plotting**: `EviDDIE/eviddie_plot_figure4.py`
- **Data**: `EviDDIE/results/ablation_curves.csv`
- Curves from validation events only; held-out events evaluated after checkpoint selection
- EviDDIE uses standard TransformerConv (no MME pharmacophore gating, no ACI KG aggregation)
- Semantic encoder: BioSentVec (700-dim), embeddings in `EviDDIE/dataset*/event_embedding2.json`

---

## Supplementary Figures

- **S1, S2**: PharDDIE component ablation (1-shot and 5-shot; MME / ACI / SRAE removed)
- **S3**: Fusion weight $\lambda$ sensitivity; $\lambda=0.3$ (i.e., 0.7:0.3 ratio) selected on validation

---

## Architecture ↔ Code Mapping

### PharDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| MME | `pharddie_layers.py`:`PharmacophoreAwareTransformerConv` | 5 pharmacophore proxies, 0.7:0.3 gating, 1+1.5$\gamma$ scaling |
| ACI | `pharddie_matcher.py`:`EmbedMatcher.neighbor_encoder()` | Bilinear attention, $\delta=z_i-z_j$, residual gating |
| SRAE | `pharddie_matcher.py`:`VAE` | $\eta=10^{-2}/10^{-3}$, latent dim=64, MSE reconstruction |
| Mol. encoder | `pharddie_models.py`:`MVN_DDI` | `PharmacophoreAwareTransformerConv` + SAGPooling |
| Pair scorer | `pharddie_matcher.py`:`EmbedMatcher.forward()` | $|z_s-z_q| \to$ MLP $\to$ sigmoid |
| Match loss | `pharddie_trainer.py`:`SigmoidLoss` | Sigmoid cross-entropy |

### EviDDIE
| Module | File:Class | Key Detail |
|--------|-----------|------------|
| Drug encoder | `eviddie_models.py`:`MVN_DDI_Block` | Standard `TransformerConv` (no MME, no KG aggregation) |
| BSA Generator | `eviddie_matcher.py`:`Generate_Model` | $700\to256\to512\to64$, Tanh |
| BSA Critic | `eviddie_matcher.py`:`Distinguish_Model` | $64\to512\to256\to128\to1$, Sigmoid |
| EVI | `eviddie_matcher.py`:`EmbedMatcher.forward()` | Softplus $\to \alpha=e+1$, EDL loss |
| SRAE | `eviddie_matcher.py`:`VAE` | Same architecture as PharDDIE SRAE |
| Comparator | `eviddie_matcher.py` | Softplus($|p_t - z_q|$) $\to$ MLP $\to \mathbb{R}^2_{\ge 0}$ |

---

## Code Fix Records

| Date | File | Fix |
|------|------|-----|
| 2026-08-04 | `PharDDIE/pharddie_layers.py` | **M2**: Pharmacophore indices corrected (N,O,C,aromatic,charge) |
| 2026-08-04 | `EviDDIE/eviddie_layers.py` | **M2**: Same correction in EviDDIE copy |
| 2026-08-04 | `PharDDIE/pharddie_trainer.py` | **M3**: SRAE `loss2_p`/`loss2_n` separation, symmetric averaging |
| 2026-08-04 | `EviDDIE/eviddie_matcher.py` | **M4**: Removed EMA prototype alignment dead code |
| 2026-08-04 | `EviDDIE/eviddie_trainer.py` | **M4**: Removed proto_loss/var_loss dead code |
| 2026-08-04 | `shared/checkpoint.py` | **M6**: `load_state_dict_safe()` with audit logging |
| 2026-08-04 | `shared/neg_manifest.py` | **M7**: SHA256-audited negative manifest generation |
| 2026-08-04 | `PharDDIE/pharddie_table4_paper.py` | **M8**: Matched-coverage selective risk comparison |
| 2026-08-08 | `EviDDIE/eviddie_matcher.py` | `Generate_Model` default dim 768$\to$700 (BioSentVec) |
| 2026-08-08 | `EviDDIE/eviddie_models.py` | Removed unused `PharmacophoreAwareTransformerConv` import |
| 2026-08-08 | `EviDDIE/eviddie_eval_ablation.py` | Fixed `IndentationError` |
| 2026-08-08 | `EviDDIE/eviddie_verify_ckpt.py` | Fixed indentation in checkpoint loop |
| 2026-08-08 | `PharDDIE/pharddie_matcher.py` | Removed `.cuda()` hardcoding; PAD mask in ACI softmax |
| 2026-08-10 | `EviDDIE/eviddie_export_zs_v2.py` | Made v2 canonical (fixed manifests); removed `_v2` suffix from output CSV |
| 2026-08-10 | `EviDDIE/eviddie_export_zs.py` | Deprecated (dynamic sampling); redirects to v2 |
| 2026-08-10 | `PharDDIE/pharddie_table3.py` | Replaced with `pharddie_table3_complete.py` (canonical); old -> `_simple.py` |
| 2026-08-10 | `EviDDIE/eviddie_case_study.py` | Fixed DrugBank evidence loader: now loads event-specific DDI records from task JSONs + e1rel_e2 |
| 2026-08-10 | `PharDDIE/pharddie_export_full.py` | Multi-training-seed support: TRAINING_SEEDS + fixed EVAL_MANIFEST_SEED |
| 2026-08-10 | `PharDDIE/pharddie_table2.py` | Group by `train_seed` when available; clarified +/- semantics in notes |
| 2026-08-10 | `PharDDIE/pharddie_table4_paper.py` | Added matched-coverage risk-coverage analysis + entropy clarification |
| 2026-08-10 | `EviDDIE/eviddie_table_discrimination.py` | **NEW**: Zero-shot discrimination table (AUC/ACC/F1) for EviDDIE |
| 2026-08-10 | `shared/audit_logger.py` | **NEW**: Audit trail utility with SHA256 + git commit logging |
| 2026-08-10 | `shared/audit_drug_overlap.py` | **NEW**: Drug overlap audit between train/dev/test splits |
| 2026-08-10 | `reproduce.ps1` | **NEW**: Full reproduction pipeline script |
| 2026-08-10 | `audit/` | **NEW**: Audit directory structure with logs, predictions, manifests |
| 2026-08-10 | `fyx_8_8(4) (1).tex` | Paper restructured around RQ1-RQ3; new EviDDIE disc. table; updated Discussion |

---

## Large Files Not in Repository

| File | Approx. Size | How to Obtain |
|------|-------------|---------------|
| `DRKG_TransE_entity.npy` | ~200 MB | Regenerate via DRKG training, or contact authors |
| `morgan_dataset*.npz` | ~50 MB each | Regenerate via `PharDDIE/fp/save_features.py` |
| Trained checkpoints (`.pth`) | ~200 MB each | Retrain with provided scripts, or contact authors |
| `ablation_curves.csv` | <1 MB | Generated by `EviDDIE/eviddie_train_ablation.py` |
| Prediction CSVs | <10 MB | Generated by respective export scripts |

---

## Known Limitations

1. **Checkpoint availability**: Trained checkpoints not in repo due to size; can be regenerated (~8-12h on RTX 4090). Per-training-seed checkpoints (5 seeds x 3 shots = 15 checkpoints) require ~15-20h total on RTX 4090.
2. **Training-seed results**: Paper now supports five independent training seeds. Table 2 +/- values represent training-seed std when per-seed checkpoints are available, or negative-sampling std otherwise (noted in table caption).
3. **EviDDIE encoder**: Uses standard `TransformerConv` (no MME/ACI), by design for zero-shot setting.
4. **Table 3 SD semantics**: SDs capture negative-sampling variation (one checkpoint) unless re-generated with per-training-seed checkpoints.
5. **Case study**: Retrospective analysis of held-out positives, not independent external validation. DrugBank evidence now verified at event level (drug pair + event type).
6. **DRKG entity embeddings**: Not included (file size). Relation embeddings are provided for symbol initialization. DRKG entity-overlap audit pending.
7. **BioSentVec weights**: Download separately from [BioSentVec repository](https://github.com/ncbi-nlp/BioSentVec). Pre-computed 700-dim embeddings are in `event_embedding2.json`.
8. **Drug overlap**: Most test-set drugs appear in training set. Claims limited to "unseen-event generalization," not "novel-compound generalization." See `shared/audit_drug_overlap.py`.
