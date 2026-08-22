# EviDDIE Dataset-2 Retrained Export — Launch

## Full run (after ALL 5 seeds finish training)

Run from the repo root (`D:\PharDDIE and EviDDIE\PharDDIE_github_8_10`):

```
C:/Users/Admin/.conda/envs/PharDDIE/python.exe external/eviddie_export_ds2.py --seeds 19940419,20230801,20240115,20240520,20240910
```

One line:

```
C:/Users/Admin/.conda/envs/PharDDIE/python.exe external/eviddie_export_ds2.py --seeds 19940419,20230801,20240115,20240520,20240910
```

Defaults (no flags needed beyond `--seeds`): dataset `EviDDIE/dataset2`, semantic
`EviDDIE/dataset2/event_embedding2.json`, checkpoints
`EviDDIE/models/eviddie_ds2_seed{seed}bestmodel` (+`_G`), tier label `test2`,
output `external/outputs/predictions_ds2_retrained_0shot.csv`.

Expected output: ~18,720 rows (25 test2 events x ~750 triples x 5 train seeds;
the T12 run on the same query set produced 3,744 rows/seed).

## Pre-flight gate

Wait until all 5 checkpoint files exist, e.g.:

```
EviDDIE/models/eviddie_ds2_seed20240910bestmodel        (+ _G, _D, _meta.json)
```

Missing checkpoints raise `FileNotFoundError` naming the seed (no partial output:
the per-seed checkpoint is resolved before any row is written for that seed).

## Smoke test (already validated 2026-08-22, seed 19940419)

```
C:/Users/Admin/.conda/envs/PharDDIE/python.exe external/eviddie_export_ds2.py --seeds 19940419 --smoke-events 2 --out_csv predictions_ds2_retrained_0shot_smoke.csv
```

## Notes

- GPU (cuda) is used if available; keep it to one run at a time to avoid
  contention with training jobs.
- The pipeline is deterministic at the evidence-chain level (row order, seeds,
  checkpoint/manifest/embedding SHA256, y_true); floating-point outputs are
  reproducible to ~1e-7 (pre-existing property of the reviewed clone's GPU
  path, verified against the T12 wrapper).
- Episode-manifest byproducts are archived as
  `external/outputs/episode_manifests/episode_manifest_ds2_retrained_0shot_seed{seed}.json`.
