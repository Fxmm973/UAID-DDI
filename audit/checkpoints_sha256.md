# Checkpoint SHA256 Manifest

Integrity record for the per-training-seed checkpoints used to produce
Table 2 and the PharDDIE rows of Table 3 (few-shot), and the zero-shot
rows of Table 3 (EviDDIE). Binary checkpoints are not stored in this
repository (16-33 MB each); they are regenerable with the provided
training scripts or available from the authors. The SHA256 values below
allow independent verification of the five-seed evidence chain.

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

- seed 19940419: main `56606b15d2b731af30344a33ca6a11986ff650ae53432c98fbcf22f64c67d7e6` (16445317 bytes); generator `06ff84c1718d58cc52a84fca73fc72fb3c3e8ff6de7a422991a0cd0c4642a3f3` (1379447 bytes)
- seed 20230801: main `fe0b1e9b4160439ea16a4c1977115a5d1be23db6f9bb65e508e413498830e2ee` (16445317 bytes); generator `f20cf3c2f3c429563da215f80fafa76cf3c9bedd8d14ca22189fd7b1783b02d1` (1379447 bytes)
- seed 20240115: main `d7bc70db46c9b861d2aaa0fe4040a7b19be4faa14939df384d353558c5f7720d` (16445317 bytes); generator `ce5b5f9bd81e833c936a2197a0ded2994f300be880ec5a6c27a45f7def21f328` (1379447 bytes)
- seed 20240520: main `cef85baa2df0893bc2964a3d3f36b9277e37b3c2f393a1c5de9d6dd48b09782c` (16445317 bytes); generator `1e32659680926664cb1917af6363a468888ab3e528488320fb4483b48f227380` (1379447 bytes)
- seed 20240910: main `f35e02ba46327c8eeae98a7dc4c30e07cf11ace88ec8cff9f275a5102ef402a0` (16445317 bytes); generator `3c5c5538a3d2ebc7efe9ef463e93476ef413646e4adb72165e00f367aeea5757` (1379447 bytes)

## BioSentVec Event Embedding Files (precomputed; keys are the event description templates)

- `EviDDIE/dataset1/event_embedding2.json`: `5f343b64a720eaa6ba6413a6a8b52edc3315f08528ee16460a0d7766b4e6631c` (1386481 bytes)
- `EviDDIE/dataset2/event_embedding2.json`: `b45bfe49a9c1a87a2c90e3878e8754a9819f8bde4717653d6fb44fa920ff1e90` (1583492 bytes)
