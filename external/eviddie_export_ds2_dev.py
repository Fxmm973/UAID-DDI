#!/usr/bin/env python
# coding=utf-8
"""Task 18: EviDDIE Dataset-2 DEV-split zero-shot export wrapper (TempScale fit set).

Dev-split twin of the T14-reviewed external/eviddie_export_ds2.py: exports
EviDDIE zero-shot predictions on EviDDIE/dataset2/dev_tasks.json (5 events)
with the deterministic dev negative manifest the T13 training wrapper
generated once under external/outputs/train_ds2_dev_manifest/ (its sha256 is
fixed in every healthy EviDDIE/models/eviddie_ds2_seed{seed}bestmodel_meta.json
and verified here per checkpoint). The dev predictions feed the Task 18
temperature-scaling fit; the T must never be fitted on test2 itself.

The T14 wrapper is imported (never edited) and layered over:

  (a) tier label -> 'dev': re-patches ExportVariantsExt.__init__ after the
      T14 patch (which forces 'test2'), so the CSV `tier` column says 'dev'.
  (b) negative manifests: after the clone's test2 manifest verification
      (EviDDIE/dataset2/neg_manifests has no dev_seed* files), the instance's
      neg_manifests is replaced with the sha-verified dev manifest.
  (c) ExportVariantsExt.export copied verbatim from the clone with three
      documented changes:
        * test_tasks read from dev_tasks.json (not test2_tasks.json);
        * pt_rel_ids built locally from the dev event keys (the clone's
          main() builds them from test2_tasks.json; that value is unused);
        * setting = 'common' (the clone hardcodes 'rare'; the v2 semantics
          map dev -> common, so calibration_table --dev-setting common
          selects these rows).
  (d) --smoke-events expands dev_tasks.json events (the T14 version reads
      test2_tasks.json); out_csv default predictions_ds2_dev_0shot.csv.
  (e) episode-manifest byproducts renamed to
      episode_manifest_ds2_dev_0shot_seed{seed}.json -- the clone writes
      episode_manifest_ext_5shot_0shot_seed{seed}.json (tier derived from
      the dataset path), which collides with git-tracked Task 8 archives.
      The rename is content-gated exactly as in T14 (episodes keyed
      'test2:<event text>' with DrugBank-id drugs) and the tracked Task 8
      archives are then restored via `git checkout --`.

Guards carried over from the T14 wrapper: --variants other than full_evi are
rejected (ds2 retrained checkpoints carry only the native two-class
evidential head); --native is rejected (dataset2 manifests have no native
entries); the semantic embedding file must exist (no placeholder fallback).

Usage (from the repo root, after all 5 ds2 retrained checkpoints exist):
  python external/eviddie_export_ds2_dev.py \
      --seeds 19940419,20230801,20240520,20260201,20260301
Smoke test (1 seed, 1 event):
  python external/eviddie_export_ds2_dev.py --seeds 19940419 --smoke-events 1 \
      --out_csv predictions_ds2_dev_0shot_smoke.csv
"""
import glob
import hashlib
import json
import logging
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

import eviddie_export_ds2 as ds2  # T14-reviewed ds2 wrapper (never edited)
import eviddie_export_ext as base  # the T8-reviewed clone
from eviddie_dataloader import DrugDataset, DrugDataLoader  # noqa: E402
from eviddie_export_ext import CLASS_ORDER, SEED_PREFIX  # noqa: E402

# ---------------------------------------------------------------------------
# (b) deterministic dev negative manifest (external/ dir; sha in model meta)
# ---------------------------------------------------------------------------
_DEV_MANIFEST_DIR = os.path.join(base._REPO, 'external', 'outputs',
                                 'train_ds2_dev_manifest')
_DEV_MANIFEST = os.path.join(_DEV_MANIFEST_DIR, 'dev_seed19940419_negatives.json')
_DEV_MANIFEST_SEED = 19940419  # fixed eval-manifest seed, as in the clone

_ORIG_INIT = base.ExportVariantsExt.__init__  # ds2._patched_init (tier='test2')


def _dev_patched_init(self, arg, ext_ent2ids):
    """(a)+(b): after the T14 init patch, label tier 'dev' and swap the
    test2 negative manifests for the sha-verified deterministic dev
    manifest (the dev-seed negatives exist only under external/)."""
    _ORIG_INIT(self, arg, ext_ent2ids)
    self.tier = 'dev'
    if not os.path.exists(_DEV_MANIFEST):
        raise FileNotFoundError(
            f'dev manifest not found: {_DEV_MANIFEST} '
            f'(the T13 training wrapper generated it once under external/; '
            f're-run its manifest-generation step if missing)')
    dev_manifest = json.load(open(_DEV_MANIFEST, encoding='utf-8'))
    actual = hashlib.sha256(open(_DEV_MANIFEST, 'rb').read()).hexdigest()
    meta_path = f'{arg.pretrained_model}_meta.json'
    if os.path.exists(meta_path):
        recorded = json.load(open(meta_path, encoding='utf-8')).get(
            'dev_manifest_sha256')
        if recorded and recorded != actual:
            raise RuntimeError(
                f'dev manifest sha256 mismatch: model meta records {recorded} '
                f'vs actual {actual}')
    self.neg_manifests = {_DEV_MANIFEST_SEED: dev_manifest}
    self.manifest_hashes = {_DEV_MANIFEST_SEED: actual}
    logging.info('[eviddie_export_ds2_dev] dev manifest loaded: %d events, '
                 'sha256=%s...', len(dev_manifest), actual[:12])


base.ExportVariantsExt.__init__ = _dev_patched_init

# ---------------------------------------------------------------------------
# (c) export() -- verbatim clone copy with the three documented changes
# ---------------------------------------------------------------------------


def _dev_export(self, mode, csv_writer, train_seed, eval_seed, method_name,
                variant, pt_rel_ids, drop_stats, event_filter=None):
    """Clone's ExportVariantsExt.export, copied verbatim, with:
    * dev_tasks.json instead of test2_tasks.json;
    * setting 'common' instead of 'rare' (v2 semantics: dev -> common);
    * rel ids built from the dev event keys (the passed pt_rel_ids was built
      by the clone's main() from test2_tasks.json and is unused here)."""
    setting = 'common'
    logging.info(f'[{method_name}] {mode.upper()} (train_seed={train_seed})')
    test_tasks = json.load(open(self.dataset + '/dev_tasks.json', encoding='utf-8'))
    dev_rel_ids = base.build_pt_rel_ids(list(test_tasks.keys()))

    neg_manifest = self.neg_manifests[eval_seed]

    with torch.no_grad():
        for query_ in test_tasks.keys():
            if event_filter is not None and query_ not in event_filter:
                continue
            query_triples = test_tasks[query_][0:]  # few=0 for zero-shot
            if not query_triples:
                continue

            manifest_entries = neg_manifest.get(query_, [])
            if len(manifest_entries) != len(query_triples):
                logging.warning(f'{query_}: manifest has {len(manifest_entries)} negs but '
                                f'{len(query_triples)} queries, skipping')
                drop_stats['manifest_mismatch'] += 1
                continue

            false_triples = []
            for entry in manifest_entries:
                d_i, d_j, d_k, rel = entry
                false_triples.append([d_i, rel, d_k])

            # ---- encodability filter (aligned: keep (query_i, neg_i) pairs) ----
            # The molecule side (DrugDataset) needs registry keys: mapped
            # drugs resolve IK14 -> DB id via ik14_to_db; new drugs were
            # registered under their IK14 from the ext drug_smiles.csv.
            # The model side (ent2id / connections / CSV ids) keeps the
            # original IK14 identifiers, which are direct ent2ids keys.
            kept_q = []   # registry-keyed positives (DrugDataset input)
            kept_f = []   # registry-keyed negatives (DrugDataset input)
            model_q = []  # original IK14 positives (ent2id / CSV)
            model_f = []  # original IK14 negatives (ent2id / CSV)
            for qt, ft in zip(query_triples, false_triples):
                res = [self.resolve_registry_key(x) for x in (qt[0], qt[2], ft[0], ft[2])]
                if (all(x in self.ent2id for x in (qt[0], qt[2], ft[0], ft[2]))
                        and all(r is not None for r in res)):
                    kept_q.append([res[0], qt[1], res[1]])
                    kept_f.append([res[2], ft[1], res[3]])
                    model_q.append(qt)
                    model_f.append(ft)
                else:
                    drop_stats['dropped_pairs'] += 1
            if not kept_q:
                logging.warning(f'{query_}: all {len(query_triples)} triples dropped '
                                f'(unencodable drugs), skipping event')
                drop_stats['empty_events'] += 1
                continue
            query_triples, false_triples = model_q, model_f

            self.episode_manifest[f'{mode}:{query_}'] = {
                'query_positives': [list(x) for x in query_triples],
                'query_negatives': [list(x) for x in false_triples],
            }

            all_triples = query_triples + false_triples
            all_rel2id = [[t[0], t[2], dev_rel_ids[t[1]]] for t in kept_q + kept_f]
            q_left = [self.ent2id[t[0]] for t in all_triples]
            q_right = [self.ent2id[t[2]] for t in all_triples]
            q_meta = self.get_meta(q_left, q_right)
            n_pos = len(query_triples)

            qb = DrugDataset(all_rel2id)
            if len(qb) != len(all_rel2id):
                raise RuntimeError(
                    f'{query_}: DrugDataset filtered {len(all_rel2id) - len(qb)} triples; '
                    f'this must not happen after the encodability filter')
            qbl = DrugDataLoader(qb, batch_size=len(all_rel2id), shuffle=False)
            qb_data = [t.to(self.device) for t in next(iter(qbl))]
            if variant == 'wo_BSA':
                if not self.linear_proj_loaded:
                    head_dir = f'models/ablation_{SEED_PREFIX[train_seed]}_seed{train_seed}'
                    self.load_linear_proj(head_dir=head_dir)
                sem = self.task_ebmedding[self.task2id[query_]].reshape(1, -1)
                task_emb = self.linear_proj(sem).detach()
            else:
                task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()

            ql_, qr_ = self.matcher.model(qb_data)
            ql = self.matcher.neighbor_encoder(q_meta[0], q_meta[1], ql_, qr_ - ql_)
            qr = self.matcher.neighbor_encoder(q_meta[2], q_meta[3], qr_, qr_ - ql_)
            qn = torch.cat((ql, qr), dim=-1)
            _, _, _, zq = self.matcher.vaemodel(qn, is_support=False, is_eval=True)
            fc_out = self.matcher.fc(torch.abs(task_emb.expand_as(zq) - zq))

            if variant == 'softmax':
                probs = F.softmax(fc_out, dim=1)[:, 1]
                unc = 1.0 - torch.max(F.softmax(fc_out, dim=1), dim=1)[0]
                ev0_np = np.zeros(probs.shape[0], dtype=np.float32)
                ev1_np = np.zeros(probs.shape[0], dtype=np.float32)
            else:
                evidence = F.softplus(fc_out)
                alpha = evidence + 1
                prob = alpha / alpha.sum(dim=1, keepdim=True)
                probs = prob[:, 1]
                unc = 2.0 / alpha.sum(dim=1)
                ev0_np = evidence[:, 0].cpu().numpy()
                ev1_np = evidence[:, 1].cpu().numpy()

            probs_np = probs.cpu().numpy()
            unc_np = unc.cpu().numpy()
            gt = np.concatenate([np.ones(n_pos), np.zeros(len(all_triples) - n_pos)])

            assert CLASS_ORDER == ('negative', 'positive'), 'class order convention changed'
            assert int(gt[:n_pos].sum()) == n_pos, 'positive labels must fill the first n_pos rows'
            assert int(gt[n_pos:].sum()) == 0, 'negative labels must fill the remaining rows'
            assert np.all((probs_np >= 0.0) & (probs_np <= 1.0)), 'probs out of [0,1]'
            assert len(probs_np) == len(all_triples), 'row count mismatch after filtering'

            rm = getattr(self, 'run_meta', {})
            for idx, (t, p, u) in enumerate(zip(all_triples, probs_np, unc_np)):
                csv_writer.writerow([rm.get('run_id', ''), train_seed, eval_seed, setting,
                                     getattr(self, 'tier', ''), 0,
                                     method_name, query_, t[0], t[2], int(gt[idx]),
                                     1 if p >= 0.5 else 0,
                                     round(float(p), 8), round(float(u), 8),
                                     round(float(ev0_np[idx]), 8), round(float(ev1_np[idx]), 8),
                                     rm.get('checkpoint_sha256', ''),
                                     rm.get('eval_manifest_sha256', ''),
                                     rm.get('event_embedding_sha256', ''),
                                     rm.get('git_commit', '')])


base.ExportVariantsExt.export = _dev_export

# ---------------------------------------------------------------------------
# (d) CLI defaults: dev out_csv; smoke expands dev_tasks.json events
# ---------------------------------------------------------------------------
_DEFAULT_OUT_CSV = 'predictions_ds2_dev_0shot.csv'


def _prepare_argv_dev(argv):
    argv = list(argv)
    ds2._inject_default(argv, '--dataset', ds2._DEFAULT_DATASET)
    ds2._inject_default(argv, '--semantic', ds2._DEFAULT_SEMANTIC)
    ds2._inject_default(argv, '--out_csv', _DEFAULT_OUT_CSV)

    # ---- semantic file must exist (no placeholder fallback, as in T14) ----
    semantic_arg = argv[argv.index('--semantic') + 1]
    if not os.path.exists(os.path.join(base._REPO, semantic_arg)):
        raise FileNotFoundError(
            f'event embedding file not found: {semantic_arg} '
            f'(refusing the clone\'s placeholder fallback for the dev export)')

    # ---- --smoke-events N: expand to the first N sorted dev events ----
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
        tasks = json.load(open(os.path.join(dataset_dir, 'dev_tasks.json'),
                               encoding='utf-8'))
        events = sorted(tasks)[:smoke]
        del argv[i:i + 2]
        ds2._inject_default(argv, '--events', ','.join(events))
        logging.info('[eviddie_export_ds2_dev] --smoke-events %d -> %d events: %s',
                     smoke, len(events), ', '.join(events))

    # ---- guards (carried over from the T14 wrapper) ----
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
# (e) episode-manifest byproducts: dev naming, restore tracked archives
# ---------------------------------------------------------------------------


def _fix_episode_manifests():
    """Move THIS run's dev episode-manifest byproducts to the ds2-dev naming,
    then restore any git-tracked archives the clone overwrote."""
    em_dir = os.path.join(base._REPO, 'external', 'outputs', 'episode_manifests')
    for p in sorted(glob.glob(os.path.join(em_dir, 'episode_manifest_ext_5shot_0shot_seed*.json'))):
        payload = json.load(open(p, encoding='utf-8'))
        if not ds2._is_dataset2_byproduct(payload):
            continue  # a Task 8 artifact; leave untouched
        payload['tier'] = 'dev'
        dst = os.path.join(em_dir,
                           f'episode_manifest_ds2_dev_0shot_seed{payload["train_seed"]}.json')
        json.dump(payload, open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'[eviddie_export_ds2_dev] episode manifest -> {os.path.basename(dst)} '
              f'({payload["n_episodes"]} episodes, payload tier set to dev)')
    subprocess.run(['git', 'checkout', '--', em_dir],
                   cwd=base._REPO, check=True, capture_output=True)


if __name__ == '__main__':
    # same entry point as the T14 wrapper: clone main + byproduct rename
    base.main(ds2._prepare_argv(_prepare_argv_dev(sys.argv[1:])))
    _fix_episode_manifests()
