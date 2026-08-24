# Checkpoint SHA256 Manifest

Integrity record for the per-training-seed checkpoints behind the shipped
per-sample prediction CSVs: `results/predictions/predictions_dataset1_PharDDIE.csv`
(PharDDIE, 5 training seeds; the source of the paper's Table 2 rows and the
PharDDIE rows of Table 3) and the EviDDIE per-seed checkpoints recorded during
the P0-7 protocol. The paper's Table 3 zero-shot rows come from five
independently trained checkpoints
(seeds 19940419, 20230801, 20240115, 20240520, 20240910), each evaluated on
the one fixed negative-sampling manifest (eval seed 19940419) — see
`EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` and
RESULTS_MAP.md. Binary checkpoints are not stored in this repository (16-33 MB
each); they are regenerable with the provided training scripts or available from
the authors. The SHA256 values below allow independent verification of the
five-seed evidence chain.

Fixed evaluation manifest seed: 19940419 for all rows. Manifest SHA256
values are recorded in `PharDDIE/dataset1/neg_manifests/manifest_hashes.json`
and `EviDDIE/neg_manifests/manifest_hashes.json`.


## PharDDIE - 1-shot (5 independent training seeds)

- seed 19940419: `5aefeb4c67912fd8dcd5bf0a5caedb8ed2fd82b76fac6c00e058db29fd4ff141` (25620293 bytes)
- seed 20230801: `762435495db2f4791242f2b943283227227848fd6eecc7429f40892b26a8ce1e` (25620293 bytes)
- seed 20240115: `e697623a0010649d21b00c2fa740a016269e68fa1ba78ddcfe7eb09eeb521f88` (25620293 bytes)
- seed 20240520: `41cb341e90b8e0355c632736f012558574f5fb16dfa717158a588982fad9f65f` (25620293 bytes)
- seed 20240910: `c37b8d0bc6552e1ffddc448f4b1fbb4d648aa75c05e0adde787e317d270bc82d` (25620293 bytes)

## PharDDIE - 5-shot (5 independent training seeds)

- seed 19940419: `ebc08a944252868233162df3ebcd1cb2c48baadf0cf886254ad6a65ff4886930` (33078890 bytes)
- seed 20230801: `7abd265439a12c165f9d19e1fd8e688f03befaed410c8472eaf43e229840a884` (25620293 bytes)
- seed 20240115: `99fef945e6aab1d3a84b69000fe80ee0b8059c1cfc723d078564a4a7995445bf` (25620293 bytes)
- seed 20240520: `0e24da77786e60b284667d75fef9603b04cac58ac30f682d516d0017ce0813b9` (25620293 bytes)
- seed 20240910: `d4a5ae672483256387dd692ade731e92b96e2ef35a61b5e0a959722bda667b28` (25620293 bytes)

## PharDDIE - 10-shot (NOT used in the paper; only seed 19940419 has a checkpoint)

- seed 19940419: `a2948f20a6b5f295fea9e84558b8688afa37b572ba3e9a525efa2c979e94062f` (33078892 bytes)
- seeds 20230801/20240115/20240520/20240910: checkpoint directories empty (no per-seed training).
  Since 10-shot is not reported in the paper, this does not affect any paper claim.

## EviDDIE - 0-shot (5 independent training seeds)

Files: `EviDDIE/models/eviddie_new_s{1..5}_seed{seed}bestmodel` (main) and
`...bestmodel_G` (generator). Main hashes equal the `checkpoint_sha256` column of
`EviDDIE/results/predictions/predictions_eviddie_new_ablation.csv` (recomputed
from disk, byte-identical).

- seed 19940419: main `4cdcec8f60523ffbf1f1ab19740b49fe7451b5d787c2eecf1247c862f18f3b71` (25519841 bytes); generator `bec50cfdad55eebeada9bea1dad48907b2e216333d842ee7fa262d21d02745bb` (1379727 bytes)
- seed 20230801: main `7430a7d56015e8640e10e7ccf39f6d8eae32ea989ea18f3b94e3cc3a34990704` (25519841 bytes); generator `1b088af16fa350db58a0d576933acf93fee352dd8fba150348edaa3e76d07fe1` (1379727 bytes)
- seed 20240115: main `b3b99958a3543c6eaa6bdf1a399a8fefa464e2ba49363a9699730e2ede71abb6` (25519841 bytes); generator `636eaa7a49c089583186f52a07f0b4faa6f32de5de02672bbf64594a77f81f45` (1379727 bytes)
- seed 20240520: main `db92c37ae53be19c5b2edbfdbe40e92e40d38509a61e435fb4b8e6b790c97178` (25519841 bytes); generator `992d78ad275bbba86160141a86839d55b17eb932139fe214c08a0c238a549efb` (1379727 bytes)
- seed 20240910: main `e118b67129259ad98a30660368b5f2ba4ebb4d1a85029e580e3cddde056a03df` (25519841 bytes); generator `a5084a0acc228444152996977e4a47e103d0e53057c359bc09d191a9a065e472` (1379727 bytes)

## BioSentVec Event Embedding Files (precomputed; keys are the event description templates)

- `EviDDIE/dataset1/event_embedding2.json`: `5f343b64a720eaa6ba6413a6a8b52edc3315f08528ee16460a0d7766b4e6631c` (1386481 bytes)
