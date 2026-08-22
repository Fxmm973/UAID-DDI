#!/usr/bin/env python
# coding=utf-8
"""Task 12 (Step 3): dataset2 cross-benchmark zero-shot export wrapper.

The reviewed clone eviddie_export_ext.py derives the CSV `tier` value from the
dataset path ('1shot' when the path contains '1shot', else '5shot'); the
dataset2 cross-benchmark tier must be labelled 'test2'. Per controller ruling
(d) the reviewed clone is never edited; this thin wrapper instead:

(a) monkeypatches ExportVariantsExt.__init__ to force self.tier = 'test2'
    (the value written into every CSV row's `tier` column), and
(b) after the run, renames THIS RUN's episode-manifest byproduct from the
    clone's derived name (episode_manifest_ext_5shot_0shot_seed*.json) to the
    dataset2 naming (episode_manifest_ext_test2_0shot_seed*.json) and fixes
    the payload's `tier` field. The rename is content-gated: only payloads
    whose episodes are dataset2 byproducts (event-text keys, DrugBank-id
    drugs) are renamed; the clone writes over the SAME filename that the
    Task 8 dataset_ext_5shot runs archived, so after renaming this run's
    files away, git restores the tracked Task 8 artifacts.

Everything else (--dataset, --semantic, --seeds, --events, --out_csv,
--variants, --native) is passed straight through to the clone's CLI.

Usage (from the repo root):
  python external/export_dataset2.py --dataset EviDDIE/dataset2 \
      --semantic EviDDIE/dataset2/event_embedding2.json \
      --seeds 19940419,20230801,20240115,20240520,20240910 \
      --out_csv predictions_dataset2_eviddie_0shot.csv
"""
import glob
import json
import os
import subprocess
import sys

import eviddie_export_ext as base

_ORIG_INIT = base.ExportVariantsExt.__init__


def _patched_init(self, arg, ext_ent2ids):
    _ORIG_INIT(self, arg, ext_ent2ids)
    self.tier = 'test2'  # dataset2 cross-benchmark tier label for the CSV


base.ExportVariantsExt.__init__ = _patched_init


def _is_dataset2_byproduct(payload):
    """True when the payload is THIS run's byproduct: episodes keyed by event
    text ('test2:<event>') whose drugs are DrugBank ids (DB...) — the ext
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
    """Rename this run's episode-manifest byproduct to the dataset2 naming,
    then restore any git-tracked Task 8 artifact the clone overwrote."""
    em_dir = os.path.join(base._REPO, 'external', 'outputs', 'episode_manifests')
    for p in sorted(glob.glob(os.path.join(em_dir, 'episode_manifest_ext_5shot_0shot_seed*.json'))):
        payload = json.load(open(p, encoding='utf-8'))
        if not _is_dataset2_byproduct(payload):
            continue  # a Task 8 artifact; leave untouched
        payload['tier'] = 'test2'
        dst = p.replace('episode_manifest_ext_5shot_0shot', 'episode_manifest_ext_test2_0shot')
        json.dump(payload, open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
        os.remove(p)
        print(f'[export_dataset2] episode manifest -> {os.path.basename(dst)} '
              f'(payload tier set to test2)')
    subprocess.run(['git', 'checkout', '--', em_dir],
                   cwd=base._REPO, check=True, capture_output=True)


if __name__ == '__main__':
    base.main(sys.argv[1:])
    _fix_episode_manifests()
