#!/usr/bin/env python
# coding=utf-8
"""案例候选 vs Dataset 2 train/dev 任务泄漏审计。

检查外部验证案例研究选出的每个候选 (drug_a, drug_b, event) 是否出现在
EviDDIE/dataset2 的 train_tasks.json 或 dev_tasks.json 中：
  1) 药对级：无序 (drug_a, drug_b) 是否出现在 train/dev 任一三元组；
  2) 三元组级：精确 (drug, event, drug)（含反向）是否出现在 train/dev。
任一命中即判 FAIL——案例必须全部 held-out。
输出：outputs/case_leakage_audit.json（判定 + 逐候选明细 + SHA256 记录）。
"""
import argparse
import hashlib
import json
import os

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def load_tasks(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(
        OUT, "case_candidates_dataset2_per_event_v2.csv"))
    ap.add_argument("--dataset", default=os.path.join(REPO, "EviDDIE", "dataset2"))
    args = ap.parse_args()

    cand = pd.read_csv(args.candidates)
    train = load_tasks(os.path.join(args.dataset, "train_tasks.json"))
    dev = load_tasks(os.path.join(args.dataset, "dev_tasks.json"))

    train_pairs, train_triples = set(), set()
    for ev, triples in train.items():
        for h, e, t in triples:
            train_pairs.add(tuple(sorted([h, t])))
            train_triples.add((h, e, t))
            train_triples.add((t, e, h))
    dev_pairs = set()
    for ev, triples in dev.items():
        for h, e, t in triples:
            dev_pairs.add(tuple(sorted([h, t])))

    details = []
    n_pair_hits, n_triple_hits = 0, 0
    for _, r in cand.iterrows():
        pair = tuple(sorted([r["drug_a"], r["drug_b"]]))
        in_train_pair = pair in train_pairs
        in_dev_pair = pair in dev_pairs
        in_train_triple = ((r["drug_a"], r["event"], r["drug_b"]) in train_triples
                           or (r["drug_b"], r["event"], r["drug_a"]) in train_triples)
        n_pair_hits += int(in_train_pair or in_dev_pair)
        n_triple_hits += int(in_train_triple)
        details.append({
            "drug_a": r["drug_a"], "drug_b": r["drug_b"], "event": r["event"],
            "in_train_pair": bool(in_train_pair), "in_dev_pair": bool(in_dev_pair),
            "in_train_triple": bool(in_train_triple),
        })

    verdict = "PASS" if (n_pair_hits == 0 and n_triple_hits == 0) else "FAIL"
    report = {
        "verdict": verdict,
        "n_candidates": len(cand),
        "n_train_pairs": len(train_pairs),
        "n_pair_hits": n_pair_hits,
        "n_triple_hits": n_triple_hits,
        "details": details,
    }
    out_path = os.path.join(OUT, "case_leakage_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"VERDICT: {verdict} | pair_hits={n_pair_hits} triple_hits={n_triple_hits} "
          f"of {len(cand)} candidates | sha256={sha[:16]}...")


if __name__ == "__main__":
    main()
