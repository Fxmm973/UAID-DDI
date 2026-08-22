# Validation-Experiment Records — Provenance Notes

Archived records behind the paper's development-stage figures:

- `weight_sweep.csv` — SHCR proxy-weight sweep (weights 0.1 / 0.2 / 0.3;
  1/5/10-shot × common/test/test2). Source: the authors' experiment log
  (`实验数据/药效团参数/药效团图/药效团数据.xlsx`, 2026-06-10). Backs
  `fig_1shot_weight_selection.jpg` / `fig_5shot_weight_selection.jpg` and the
  paper's claim that the proxy weight 0.3 was selected on validation.
- `ablation_results.csv` — per-config ablation records (no_ACI / no_meta(SHCR) /
  no_SRAE; 1/5/10-shot × COMMON_TEST/TEST/TEST2). Source: the authors'
  experiment records `result_ph2p0_{1,5,10}shot_40k.txt` (2026-02-06..09).
  Backs `1-shot-ablation.jpg` / `5-shot-ablation.jpg`.

**Important caveats (read before citing):**

1. **Pre-unified protocol.** These records were produced by the pre-revision
   code snapshots (single training seed, `dropout` argument 0.5, max_batches 40000). 
2. **Legacy naming.** `no_meta(SHCR)` = the `--no_meta` ablation switch of the
   old codebase, which disabled the module the paper calls SHCR
   (Selected Hidden-Channel Reweighting). (The raw recorder files' project-name
   field was a legacy value; the project is PharDDIE.)
3. **Era-mismatched reference runs.** The full-model reference used when the
   figures were drawn (same-era runs as the ablation records) is not archived
   together with the ablation records; `weight_sweep.csv` is a later snapshot
   (2026-06). Comparisons across these two files are therefore indicative
   only and must not be quoted as the figure's plotted values.
4. The paper's figures remain the authoritative rendering of these records;
   the plotting scripts below reproduce the data tables, not the exact
   pixel layout of the shipped JPGs.
