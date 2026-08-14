#!/usr/bin/env python
# coding=utf-8
"""
Verify negative manifests against recorded SHA256 hashes and expected entry counts.

Run from the repository root, e.g.:
    python shared/verify_manifests.py --hash-log PharDDIE/dataset1/neg_manifests/manifest_hashes.json \
                                      --manifest-dir PharDDIE/dataset1/neg_manifests \
                                      --dataset PharDDIE/dataset1
    python shared/verify_manifests.py --hash-log EviDDIE/neg_manifests/manifest_hashes.json \
                                      --manifest-dir EviDDIE/neg_manifests \
                                      --dataset EviDDIE/dataset1

Exits 0 only if every recorded manifest exists, its SHA256 matches the recorded
value, and (when --dataset is given) the per-event entry counts match the
corresponding task JSON. Used by reproduce.ps1 before any export/table step.
"""
import json
import hashlib
import os
import sys
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hash-log', required=True)
    ap.add_argument('--manifest-dir', required=True)
    ap.add_argument('--dataset', default=None, help='optional: also check entry counts vs task JSONs')
    args = ap.parse_args()

    if not os.path.exists(args.hash_log):
        print(f'FAIL: hash log not found: {args.hash_log}')
        sys.exit(1)
    log = json.load(open(args.hash_log))

    ok = True
    for name, entry in sorted(log.items()):
        recorded = entry.get('sha256') if isinstance(entry, dict) else entry
        mf = os.path.join(args.manifest_dir, f'{name}_negatives.json')
        if not os.path.exists(mf):
            print(f'FAIL: manifest missing: {mf}')
            ok = False
            continue
        actual = hashlib.sha256(open(mf, 'rb').read()).hexdigest()
        if actual != recorded:
            print(f'FAIL: hash mismatch {mf}: recorded={recorded}, actual={actual}')
            ok = False
            continue
        print(f'OK: {name} sha256={actual[:16]}...')

        if args.dataset:
            split = name.split('_')[0]
            task_path = os.path.join(args.dataset, f'{split}_tasks.json')
            if not os.path.exists(task_path):
                print(f'FAIL: task file not found: {task_path}')
                ok = False
                continue
            tasks = json.load(open(task_path))
            manifest = json.load(open(mf))
            for evt, triples in tasks.items():
                if len(manifest.get(evt, [])) != len(triples):
                    print(f'FAIL: count mismatch {name} event={evt}: '
                          f'manifest={len(manifest.get(evt, []))} task={len(triples)}')
                    ok = False
            for evt in manifest:
                if evt not in tasks:
                    print(f'FAIL: manifest event not in task file: {evt}')
                    ok = False

    if ok:
        print('MANIFEST VERIFICATION PASSED')
    else:
        print('MANIFEST VERIFICATION FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
