# external/tests/test_neg_manifest_ext.py
import json, pytest
from neg_manifest_ext import generate_manifest_ext

TASKS = {"PT-1": [["A", "PT-1", "B"], ["A", "PT-1", "C"], ["A", "PT-1", "D"]]}
CAND = {"PT-1": ["A", "B", "C", "D", "E", "F"]}

def test_manifest_deterministic_and_no_collision():
    m1 = generate_manifest_ext(TASKS, CAND, seed=19940419)
    m2 = generate_manifest_ext(TASKS, CAND, seed=19940419)
    assert m1 == m2
    for entries in m1.values():
        for d_i, d_j, d_k, rel in entries:
            assert d_k != d_j                       # 不采样自身
            assert [d_i, rel, d_k] not in TASKS[rel]  # 不在已知阳性中

def test_manifest_entry_count_matches_tasks():
    m = generate_manifest_ext(TASKS, CAND, seed=19940419)
    for event, entries in m.items():
        assert len(entries) == len(TASKS[event])
