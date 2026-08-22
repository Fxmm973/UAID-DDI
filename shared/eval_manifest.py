#!/usr/bin/env python
# coding=utf-8
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_fixed_event_rows(dataset_dir, split, manifest_seed):
    dataset_dir = Path(dataset_dir)
    tasks_path = dataset_dir / f"{split}_tasks.json"
    manifest_path = (
        dataset_dir / "neg_manifests"
        / f"{split}_seed{manifest_seed}_negatives.json"
    )

    tasks = json.load(open(tasks_path, encoding="utf-8"))
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    rows = []
    for event, positives in tasks.items():
        entries = manifest.get(event)
        if entries is None:
            raise ValueError(f"Missing manifest event: {event}")

        neg_by_positive = {}
        for head, positive_tail, negative_tail, relation in entries:
            key = (head, relation, positive_tail)
            if key in neg_by_positive:
                raise ValueError(f"Duplicate manifest key: {key}")
            neg_by_positive[key] = negative_tail

        if len(neg_by_positive) != len(positives):
            raise ValueError(
                f"{event}: {len(positives)} positives but "
                f"{len(neg_by_positive)} manifest entries"
            )

        for head, relation, tail in positives:
            key = (head, relation, tail)
            if key not in neg_by_positive:
                raise ValueError(f"Manifest does not match positive: {key}")
            negative_tail = neg_by_positive[key]
            rows.append((event, head, relation, tail, 1))
            rows.append((event, head, relation, negative_tail, 0))

    return rows, sha256_file(manifest_path)
