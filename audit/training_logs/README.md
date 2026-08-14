# Per-Seed Training Logs

Training logs (TensorBoard tfevents / plain text), one directory per (setting, training seed).
These correspond to the five independent training runs behind the paper's $\pm$ values.

## PharDDIE (few-shot, Dataset 1)

- 1-shot: seeds 19940419, 20230801, 20240115, 20240520, 20240910 — 5/5 present.
- 5-shot: seeds 20230801, 20240115, 20240520, 20240910 — 4/5 present. The log for
  seed 19940419 was not retained on disk; its checkpoint exists and its SHA256 is
  recorded in `audit/checkpoints_sha256.md`.
- 10-shot: not used in the paper; logs not archived here.

## EviDDIE (zero-shot, Dataset 1)

- 0-shot: all five seeds present (tfevents files).

Each directory name encodes the checkpoint path loaded by the corresponding export
script (e.g., `models/dataset1/models_drugbank_1shot_str_seed19940419/bestmodel`).
