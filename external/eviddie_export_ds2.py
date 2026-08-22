#!/usr/bin/env python
# coding=utf-8
"""Task 14: EviDDIE Dataset-2 RETRAINED zero-shot export wrapper.

Same export pipeline as the reviewed clone eviddie_export_ext.py, but loading
the Dataset-2 RETRAINED checkpoints (EviDDIE/models/eviddie_ds2_seed{seed}
bestmodel / bestmodel_G) instead of the dataset1 zero-shot checkpoints the
clone hardcodes. The clone is never edited (controller ruling (d), T8); this
thin wrapper monkeypatches:

(a) base.resolve_checkpoint -> resolve under EviDDIE/models with prefix
    'eviddie_ds2' (both flat <base>/eviddie_ds2_seed<seed>bestmodel and
    directory <base>/eviddie_ds2_seed<seed>/bestmodel layouts, as in the
    clone's own resolver), so main()'s checkpoint hash chain and run_meta
    (checkpoint_sha256 / eval_manifest_sha256 / event_embedding_sha256 /
    git_commit) record the retrained artifacts untouched;
(b) ExportVariantsExt.__init__ -> force self.tier = 'test2' (the value
    written into every CSV row's `tier` column; the clone would derive the
    wrong label '5shot' from the dataset path), exactly as the T12 wrapper
    does.

Dataset defaults point at the dataset2 cross-benchmark artifacts:
  --dataset    EviDDIE/dataset2
  --semantic   EviDDIE/dataset2/event_embedding2.json   (106 prototypes,
               keyed by event text; the clone reshapes each value to 700-d)
  --out_csv    predictions_ds2_retrained_0shot.csv

CLI mirrors the clone (--seeds --events --variants --native --out_csv ...)
plus one convenience flag:
  --smoke-events N   restrict the run to the first N (sorted) test2 events
                     (expanded into the clone's --events list before main()).

Guards (evidence integrity):
  * --variants other than 'full_evi' are rejected: the ds2 retrained
    checkpoints carry only the native two-class evidential head; ablation
    heads (fc_*.pt / linear_proj_wo_BSA.pt) only exist for the dataset1
    models, so a non-default variant would silently load dataset1 ablations.
  * --native is rejected: dataset2 manifests have no native variants
    (manifest_hashes.json carries only test2_seed{seed} entries).

Episode-manifest byproducts: the clone archives them under
episode_manifest_ext_5shot_0shot_seed{seed}.json (tier derived from the
dataset path), which collides with the git-tracked Task 8 ext archives. As in
T12, this run's byproducts are moved to their own naming -- but here to
episode_manifest_ds2_retrained_0shot_seed{seed}.json, keeping the T12 test2
episode archives (dataset1-checkpoint run) untouched -- then `git checkout
--` restores the tracked Task 8 archives. The rename is content-gated on the
episodes being dataset2 byproducts (test2:<event text> keys, DrugBank-id
drugs).

Usage (from the repo root, after all 5 ds2 retrained checkpoints exist):
  python external/eviddie_export_ds2.py --seeds 19940419,20230801,20240115,20240520,20240910
Smoke test (1 seed, 2 events):
  python external/eviddie_export_ds2.py --seeds 19940419 --smoke-events 2 \
      --out_csv predictions_ds2_retrained_0shot_smoke.csv
"""
import glob
import json
import logging
import os
import subprocess
import sys

import eviddie_export_ext as base

# ---------------------------------------------------------------------------
# (a) checkpoint resolution override
# ---------------------------------------------------------------------------
_DS2_MODELS_DIR = os.path.join(base._EVIDDIE, 'models')
_DS2_PREFIX = 'eviddie_ds2'
_ORIG_RESOLVE = base.resolve_checkpoint


def _ds2_resolve_checkpoint(train_seed, prefix, base_dir):
    """Ignore the clone's dataset1 prefix/base and resolve the ds2 retrained
    checkpoint for `train_seed` (same flat/dir fallback logic)."""
    m, g = _ORIG_RESOLVE(train_seed, _DS2_PREFIX, _DS2_MODELS_DIR)
    if not os.path.exists(m):
        raise FileNotFoundError(
            f'ds2 retrained checkpoint not found for seed {train_seed}: {m} '
            f'(training may still be running)')
    return m, g


base.resolve_checkpoint = _ds2_resolve_checkpoint

# ---------------------------------------------------------------------------
# (b) tier label override (same as the T12 wrapper)
# ---------------------------------------------------------------------------
_ORIG_INIT = base.ExportVariantsExt.__init__


def _patched_init(self, arg, ext_ent2ids):
    _ORIG_INIT(self, arg, ext_ent2ids)
    self.tier = 'test2'  # dataset2 cross-benchmark tier label for the CSV


base.ExportVariantsExt.__init__ = _patched_init

# ---------------------------------------------------------------------------
# CLI default injection + guards
# ---------------------------------------------------------------------------
_DEFAULT_DATASET = 'EviDDIE/dataset2'
_DEFAULT_SEMANTIC = 'EviDDIE/dataset2/event_embedding2.json'
_DEFAULT_OUT_CSV = 'predictions_ds2_retrained_0shot.csv'


def _inject_default(argv, flag, value):
    if not any(a == flag for a in argv):
        argv += [flag, value]
        logging.info('[eviddie_export_ds2] %s -> %s (default)', flag, value)
    return argv


def _prepare_argv(argv):
    argv = list(argv)
    _inject_default(argv, '--dataset', _DEFAULT_DATASET)
    _inject_default(argv, '--semantic', _DEFAULT_SEMANTIC)
    _inject_default(argv, '--out_csv', _DEFAULT_OUT_CSV)

    # ---- the clone silently falls back to zero-vector placeholder
    # embeddings when the semantic file is missing; for a retrained-export
    # run that would misrepresent the model's task embeddings, so fail ----
    semantic_arg = argv[argv.index('--semantic') + 1]
    if not os.path.exists(os.path.join(base._REPO, semantic_arg)):
        raise FileNotFoundError(
            f'event embedding file not found: {semantic_arg} '
            f'(refusing the clone\'s placeholder fallback for the retrained export)')

    # ---- --smoke-events N: expand to the first N sorted test2 events ----
    smoke = None
    if '--smoke-events' in argv:
        i = argv.index('--smoke-events')
        try:
            smoke = int(argv[i + 1])
        except (IndexError, ValueError):
            raise RuntimeError('--smoke-events requires an integer N >= 1')
        if smoke < 1:
            raise RuntimeError('--smoke-events requires an integer N >= 1')
        dataset_arg = argv[argv.index('--dataset') + 1]
        dataset_dir = os.path.join(base._REPO, dataset_arg)
        tasks = json.load(open(os.path.join(dataset_dir, 'test2_tasks.json'),
                               encoding='utf-8'))
        events = sorted(tasks)[:smoke]
        del argv[i:i + 2]
        _inject_default(argv, '--events', ','.join(events))
        logging.info('[eviddie_export_ds2] --smoke-events %d -> %d events: %s',
                     smoke, len(events), ', '.join(events))

    # ---- guards ----
    if '--native' in argv:
        raise RuntimeError('--native is unsupported for dataset2: '
                           'manifest_hashes.json has no test2_seed{seed}_native entries')
    if '--variants' in argv:
        variants = argv[argv.index('--variants') + 1]
        if variants.strip().lower() != 'full_evi':
            raise RuntimeError(
                f'variant {variants!r} is unsupported for the ds2 retrained '
                f'checkpoints: only full_evi exists (ablation heads are '
                f'dataset1-only artifacts and would be silently mis-loaded)')
    return argv

# ---------------------------------------------------------------------------
# episode-manifest byproduct: rename this run's files, restore tracked ones
# ---------------------------------------------------------------------------


def _is_dataset2_byproduct(payload):
    """True when the payload is THIS run's byproduct: episodes keyed by event
    text ('test2:<event>') whose drugs are DrugBank ids (DB...) -- the ext
    tiers archive IK14/PT-code episodes under the same filename."""
    eps = payload.get('episodes', {})
    if not eps:
        return False
    if not all(str(k).startswith('test2:') and 'PT-' not in str(k) for k in eps):
        return False
    first = next(iter(eps.values()))
    drugs = [t[0] for t in first.get('query_positives', [])]
    if not drugs:
        return False
    return all(str(d).startswith('DB') for d in drugs)


def _fix_episode_manifests():
    """Move this run's episode-manifest byproducts to the ds2-retrained
    naming, then restore any git-tracked archives the clone overwrote."""
    em_dir = os.path.join(base._REPO, 'external', 'outputs', 'episode_manifests')
    for p in sorted(glob.glob(os.path.join(em_dir, 'episode_manifest_ext_5shot_0shot_seed*.json'))):
        payload = json.load(open(p, encoding='utf-8'))
        if not _is_dataset2_byproduct(payload):
            continue  # a Task 8 artifact; leave untouched
        payload['tier'] = 'test2'
        dst = os.path.join(em_dir,
                           f'episode_manifest_ds2_retrained_0shot_seed{payload["train_seed"]}.json')
        json.dump(payload, open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'[eviddie_export_ds2] episode manifest -> {os.path.basename(dst)} '
              f'({payload["n_episodes"]} episodes, payload tier set to test2)')
    subprocess.run(['git', 'checkout', '--', em_dir],
                   cwd=base._REPO, check=True, capture_output=True)


if __name__ == '__main__':
    base.main(_prepare_argv(sys.argv[1:]))
    _fix_episode_manifests()
