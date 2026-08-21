# external/tests/test_build_dataset_ext.py
import json, pytest
from build_dataset_ext import build_tasks, canonical_triple

def test_canonical_triple_orders_by_ik14():
    assert canonical_triple("ZDIG..", "XSBS..", "PT-1") == ["XSBS..", "PT-1", "ZDIG.."]
    assert canonical_triple("XSBS..", "ZDIG..", "PT-1") == ["XSBS..", "PT-1", "ZDIG.."]

def test_build_tasks_tiers():
    pairs = {  # (a, b, event)
        ("A", "B", "PT-1"), ("A", "C", "PT-1"),
        ("A", "B", "PT-2"), ("A", "C", "PT-2"), ("A", "D", "PT-2"),
        ("B", "C", "PT-2"), ("B", "D", "PT-2"), ("C", "D", "PT-2"),
        ("A", "B", "PT-3"),
    }
    tasks_1, tasks_5 = build_tasks(pairs, min_pairs_1shot=2, min_pairs_5shot=6)
    assert set(tasks_1) == {"PT-1", "PT-2"}
    assert set(tasks_5) == {"PT-2"}
    assert all(len(v) == 6 for v in tasks_5.values())
    # 字典序固定：每事件内按 pair_id 排序
    for event, triples in tasks_1.items():
        pair_ids = ["::".join(sorted([t[0], t[2]])) for t in triples]
        assert pair_ids == sorted(pair_ids)
