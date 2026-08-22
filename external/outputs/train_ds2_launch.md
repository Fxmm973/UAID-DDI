# EviDDIE Dataset-2 Training — Launch Guide (Task 13)

Train EviDDIE on `EviDDIE/dataset2/` (Lin-protocol benchmark) with **dataset2
molecular graphs** (the stock `shared/preprocess.py` would silently use
dataset1's graphs; `external/train_eviddie_dataset2.py` rebuilds the molecule
registry from `EviDDIE/dataset2/drug_smiles.csv` and rebinds it into the
training path before `eviddie_dataloader` is imported).

## Hyperparameters (identical to README formal training)

`--dataset EviDDIE/dataset2 --few 10 --train_few 10 --batch_size 256 --lr 1e-3
--max_batches 20000 --eval_every 1000 --log_every 50 --semantic
event_embedding2.json --semantic_noise 0.3 --embed_model TransE --aggregate max
--dropout 0.2 --fine_tune True --max_neighbor 30 --weight_decay 0.0`
(EDL annealing 10000). All of these are the `eviddie_args.py` defaults the
wrapper keeps; the wrapper only overrides dataset / prefix / seed / max_batches.

## Environment

- Python (GPU): `C:/Users/Admin/.conda/envs/PharDDIE/python.exe` (torch 2.0.1 + CUDA, RTX 4090).
- Run from **any** directory — the wrapper resolves all paths relative to the
  repo root (R7). It chdirs into `EviDDIE/` itself so checkpoints land where
  the stock trainer puts them.

## Per-seed launch (serial; 5 runs)

```bat
cd /d "D:\PharDDIE and EviDDIE\PharDDIE_github_8_10"

"C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 19940419 --max-batches 20000
"C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 20230801 --max-batches 20000
"C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 20240115 --max-batches 20000
"C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 20240520 --max-batches 20000
"C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 20240910 --max-batches 20000
```

Optional `--device-id N` pins the GPU via `CUDA_VISIBLE_DEVICES` (set before
torch import).

## 2-parallel variant (48 GB VRAM, batch 256 comfortably fits twice)

```bat
start "ds2-s1" "C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 19940419 --max-batches 20000
start "ds2-s3" "C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 20240115 --max-batches 20000
```

Then the same two-`start` pattern for (20230801, 20240520), then 20240910
alone. Two concurrent runs may add ~10–20% per-run overhead versus serial.

## Wall time

Measured on the smoke run (seed 19940419, 300 iterations): **0.54 s/iter**
steady-state on RTX 4090 (161 s for 300 iterations, 11:05:05 → 11:07:45) →
per seed ≈ **3.0 h** for 20,000 iterations. Serial 5 seeds ≈ **15 h**;
2-parallel ≈ **9–10 h** wall (each pair ≈3.3 h including contention overhead,
last seed alone ≈3.0 h).

## Outputs

| Artifact | Path (relative to repo root) |
|---|---|
| Dev-best matcher | `EviDDIE/models/eviddie_ds2_seed{seed}bestmodel` |
| Generator / critic | `EviDDIE/models/eviddie_ds2_seed{seed}bestmodel_G` / `..._D` |
| Checkpoint metadata | `EviDDIE/models/eviddie_ds2_seed{seed}bestmodel_meta.json` (train_seed, best_step, dev AUROC, dev manifest sha256) |
| Final save (20 000) | `EviDDIE/models/eviddie_ds2_seed{seed}` |
| Run log | `external/outputs/train_logs_ds2/log-eviddie_ds2_seed{seed}.txt` |
| Result record | `external/outputs/train_logs_ds2/result_eviddie_ds2_seed{seed}.txt` |
| TensorBoard | `external/outputs/train_logs_ds2/tensorboard/eviddie_ds2` |

Dev checkpoint selection uses the fixed P0-4 manifest
(`--eval-manifest-seed 19940419` default): the wrapper generates
`external/outputs/train_ds2_dev_manifest/dev_seed19940419_negatives.json` on
first run (deterministic tail-corruption, same algorithm as
`external/neg_manifest_ext.py`) because `EviDDIE/dataset2/neg_manifests/` only
ships `test2_seed*` files, and patches `eviddie_trainer.load_fixed_event_rows`
to fall back to it. Nothing is written inside `EviDDIE/dataset2/`.

## Smoke test (done, do not re-run for full training)

```bat
"C:\Users\Admin\.conda\envs\PharDDIE\python.exe" external\train_eviddie_dataset2.py --seed 19940419 --max-batches 300 --prefix eviddie_ds2_smoke
```

Evidence: 1,258/1,258 dataset2 drugs parsed and registered; 244 dataset2-only
drug ids asserted present in the graph dict; loss 1.2436 (batch 0) → 1.1131
(batch 100) → 1.1027 (batch 150) → 1.1242 (batch 250) — decreased from
initial; ROC 0.4967 → 0.5154; checkpoint written
(`EviDDIE/models/eviddie_ds2_smoke_seed19940419`, 25.5 MB). Log:
`external/outputs/train_logs_ds2/log-eviddie_ds2_smoke_seed19940419.txt`.

## Verification checklist for the controller

1. Per-seed `bestmodel_meta.json` exists with distinct `train_seed` and
   `dev_manifest_sha256 = 7ff2e86357b762e6f72cf8d86bc3b035f4cf5a349a2bab7b77205749b1040fcb`
   (dataset2 dev manifest).
2. `[DS2-GRAPHS]` line in each log: `1258 molecule graphs built`.
3. Loss log lines trend downward; dev eval every 1000 iterations.
4. Matcher/G/D checkpoints have non-identical SHA256 across seeds.
