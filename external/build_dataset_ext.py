# external/build_dataset_ext.py
"""把 RxPairEvid-50K 组装成 dataset1 同构的 dataset_ext_{1shot,5shot}。
关键不变量：ent2ids/ent2embids/relation2ids 的旧键值内容与顺序与 dataset1 完全一致
（保证 symbol2vec 前 N 行与训练时逐行相同，checkpoint 可逐位加载）。

裁决落实情况（plan: docs/superpowers/plans/2026-08-21-rxpairevid-external-validation.md）：
- R13a：ik14_to_db 非 1:1（dataset1 内 29 个 IK14 对应多个 DB ID）。新药 IK14 命中时取
  字典序最小 DB ID，并在构建报告 ik14_to_db_id_choices 中确定性记录。committed 的
  ik14_to_db.json 由 dict(zip) 生成（后值覆盖），对 29 个共享 IK14 并非字典序最小，故本
  模块从 dataset1 的 drug_smiles.csv 重建全量 {ik14: sorted(db_ids)}（方法同 Task 3：
  RDKit MolToInchiKey 前 14 位），并与 committed 文件做键集一致校验。
- R14：ik14_smiles_map.json 顶层含非 dict 条目（__meta__ 来源标注），取值时忽略之；
  药物解析一律走 IK14 键，不做跨源 SMILES 字符串比对。
- R13b：构建报告记录药对级泄漏检查：873 个信号对（无序、IK14 表示）出现在 dataset1
  train/dev/test/test2 任务文件（t[0],t[2] 无序药对）中的数量与比例。
- R15：6 个无 PubChem SMILES（no_cid / no_smiles）且不在 dataset1 的药物（1-shot 层 6 个、
  5-shot 层 3 个）——涉及这些药物的信号对在分层前剔除（不进入任何 tier 的任务、
  drug_smiles.csv、ent2ids/ent2embids），层定义阈值（≥2 / ≥6 对）保持不变，剔除记录写入
  构建报告（excluded_drugs / n_excluded_pairs / events_dropped_by_exclusion）。
  实测结果：1-shot 事件数 185 → 179，5-shot 24 → 23。理由：无分子图的药物在分子编码器下
  只能得到退化图，评分无意义（controller R15，2026-08-21）。"""
import json, os, shutil, hashlib
from collections import defaultdict

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DS1 = os.path.join(REPO, "PharDDIE", "dataset1")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_triple(a_ik14, b_ik14, event):
    a, b = sorted([a_ik14, b_ik14])
    return [a, event, b]


def build_tasks(signal_pairs, min_pairs_1shot=2, min_pairs_5shot=6):
    """signal_pairs: iterable of (a_ik14, b_ik14, event)；返回 (tasks_1, tasks_5)。"""
    by_event = defaultdict(list)
    for a, b, e in signal_pairs:
        by_event[e].append(canonical_triple(a, b, e))

    def tier(min_pairs):
        tasks = {}
        for e, triples in sorted(by_event.items()):
            if len(triples) >= min_pairs:
                pair_ids = ["::".join(sorted([t[0], t[2]])) for t in triples]
                tasks[e] = [t for _, t in sorted(zip(pair_ids, triples))]
        return tasks

    return tier(min_pairs_1shot), tier(min_pairs_5shot)


def ik14_of(smiles):
    """SMILES -> InChIKey 前 14 位（连接层），与 audit_overlap_ext.ik14_of 同方法（R13a）。"""
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol)[:14] if mol else None
    except Exception:
        return None


def build_ik14_to_db_ids(ds1_smiles_df):
    """dataset1 药物 SMILES -> {ik14: sorted([db_id, ...])}（全量映射，R13a 用）。"""
    ds1 = ds1_smiles_df.copy()
    ds1["ik14"] = ds1["smiles"].map(ik14_of)
    out = defaultdict(list)
    for r in ds1.itertuples(index=False):
        if r.ik14:
            out[r.ik14].append(r.drug_id)
    return {k: sorted(v) for k, v in out.items()}


def dataset1_task_pairs(ds1_dir=DS1):
    """dataset1 train/dev/test/test2 任务三元组 t[0],t[2] 的无序药对集合（R13b）。"""
    pairs = set()
    for fn in ("train_tasks.json", "dev_tasks.json", "test_tasks.json", "test2_tasks.json"):
        tasks = json.load(open(os.path.join(ds1_dir, fn)))
        for triples in tasks.values():
            for t in triples:
                pairs.add(tuple(sorted((t[0], t[2]))))
    return pairs


def assemble_dataset(tier_name, tasks, ik14_smiles_map, ik14_to_db_ids, ds1_smiles_df, total_events):
    """返回 build_report 条目；写 PharDDIE/dataset_ext_{tier_name}/。

    ik14_to_db_ids: {ik14: sorted([db_id, ...])}（R13a：命中时取字典序最小 DB ID 行）。"""
    dst = os.path.join(REPO, "PharDDIE", f"dataset_ext_{tier_name}")
    os.makedirs(dst, exist_ok=True)
    # 事件键集合
    events = list(tasks)
    # 药物集合（任务中出现的所有 id）
    drugs = sorted({d for triples in tasks.values() for t in triples for d in (t[0], t[2])})
    # 1) 任务与候选
    json.dump(tasks, open(os.path.join(dst, "test2_tasks.json"), "w"), indent=2, ensure_ascii=False)
    rel2candidates = {}
    for e, triples in tasks.items():
        pool = {t[0] for t in triples} | {t[2] for t in triples}
        rel2candidates[e] = sorted(pool)
    json.dump(rel2candidates, open(os.path.join(dst, "rel2candidates.json"), "w"), indent=2, ensure_ascii=False)
    # 2) drug_smiles：dataset1 原文 + 新药（IK14 或映射后的 DB ID 行）
    ds1_rows = ds1_smiles_df.to_dict("records")
    known_ids = set(ds1_smiles_df["drug_id"])
    new_rows, mapped_choices = [], {}
    for d in drugs:
        if d in known_ids:
            continue  # dataset1 已有
        db_ids = ik14_to_db_ids.get(d)
        if db_ids:
            mapped_choices[d] = db_ids[0]  # R13a：字典序最小 DB ID；SMILES 用 dataset1 的
            continue
        rec = ik14_smiles_map.get(d)
        smi = rec.get("smiles") if isinstance(rec, dict) else None  # R14：忽略 __meta__ 等非 dict 条目
        if not smi:
            # R15：无 SMILES 且无 dataset1 映射的药物已在 main 层剔除；此路径为不变量守卫
            raise ValueError(f"no SMILES for drug {d}")
        new_rows.append({"drug_id": d, "smiles": smi})
    pd.DataFrame(ds1_rows + new_rows).to_csv(os.path.join(dst, "drug_smiles.csv"), index=False)
    # 3) 符号表：旧文件原文 + 新药追加（-1）
    ent2ids = json.load(open(os.path.join(DS1, "ent2ids")))
    ent2embids = json.load(open(os.path.join(DS1, "ent2embids")))
    for d in drugs:
        if d not in ent2ids:
            ent2ids[d] = max(ent2ids.values()) + 1
            ent2embids[d] = -1
    json.dump(ent2ids, open(os.path.join(dst, "ent2ids"), "w"), indent=2)
    json.dump(ent2embids, open(os.path.join(dst, "ent2embids"), "w"), indent=2)
    # 4) 逐字节复制静态资源并记录哈希
    hashes = {}
    for f in ["relation2ids", "relation2embids", "e1rel_e2.json", "path_graph_train_only",
              "DRKG_TransE_entity.npy", "DRKG_TransE_relation.npy"]:
        shutil.copy2(os.path.join(DS1, f), os.path.join(dst, f))
        hashes[f] = {"src": sha256(os.path.join(DS1, f)), "dst": sha256(os.path.join(dst, f))}
        assert hashes[f]["src"] == hashes[f]["dst"], f"copy mismatch for {f}"
    return {"tier": tier_name, "n_events": len(events), "n_discarded_events": total_events - len(events),
            "n_drugs": len(drugs), "n_new_drugs": len(new_rows),
            "n_mapped_drugs": len(mapped_choices),
            "ik14_to_db_id_choices": mapped_choices, "hashes": hashes}


def main():
    ikmap = json.load(open(os.path.join(OUT, "ik14_smiles_map.json")))
    ik14_to_db_committed = json.load(open(os.path.join(OUT, "ik14_to_db.json")))
    df = pd.read_csv(os.path.join(REPO, "external", "raw", "ddi_pairs_50k.csv"),
                     dtype={"drug_a_ik14": "string", "drug_b_ik14": "string"})
    sig = df[df["faers_ror95_lcl_max_strict"].notnull()]
    pairs_all = [(r["drug_a_ik14"], r["drug_b_ik14"],
                  "PT-" + str(int(float(r["faers_best_pt_code_strict"]))))
                 for _, r in sig.iterrows()]
    ds1_smiles_df = pd.read_csv(os.path.join(DS1, "drug_smiles.csv"), dtype={"drug_id": "string"})
    ik14_to_db_ids = build_ik14_to_db_ids(ds1_smiles_df)
    if set(ik14_to_db_committed) != set(ik14_to_db_ids):
        raise RuntimeError(
            f"rebuilt IK14->DB map key set mismatch vs committed ik14_to_db.json "
            f"({len(ik14_to_db_committed)} vs {len(ik14_to_db_ids)})")

    def lacks_smiles(ik14):
        rec = ikmap.get(ik14)
        return not (rec.get("smiles") if isinstance(rec, dict) else None)

    # R15：无 SMILES 且无 dataset1 映射的药物在分层前剔除（阈值 ≥2 / ≥6 对不变）
    excluded_drugs = sorted({d for a, b, _ in pairs_all for d in (a, b)
                             if lacks_smiles(d) and not ik14_to_db_ids.get(d)})
    pairs = [(a, b, e) for a, b, e in pairs_all
             if a not in excluded_drugs and b not in excluded_drugs]
    n_excluded_pairs = len(pairs_all) - len(pairs)
    tasks_all_1, tasks_all_5 = build_tasks(pairs_all)  # 供 events_dropped_by_exclusion 对比
    tasks_1, tasks_5 = build_tasks(pairs)
    total_events = len({e for _, _, e in pairs_all})
    # R13b：药对级泄漏检查（按 R15 保持全部 873 个信号对）
    sig_pairs = [tuple(sorted((a, b))) for a, b, _ in pairs_all]
    ds1_pairs = dataset1_task_pairs()
    overlap = 0
    for a, b in sig_pairs:
        da, db = ik14_to_db_ids.get(a), ik14_to_db_ids.get(b)
        if not da or not db:
            continue
        if any(tuple(sorted((x, y))) in ds1_pairs for x in da for y in db):
            overlap += 1
    report = {
        "n_signal_pairs": len(sig_pairs),
        "pair_overlap_with_dataset1": overlap,
        "pair_overlap_rate": round(overlap / len(sig_pairs), 6),
        "n_pt_events_total": total_events,
        "n_excluded_drugs": len(excluded_drugs),
        "excluded_drugs": excluded_drugs,
        "n_excluded_pairs": n_excluded_pairs,
        "tiers": {},
    }
    for tier_name, tasks, tasks_all in (("1shot", tasks_1, tasks_all_1),
                                        ("5shot", tasks_5, tasks_all_5)):
        entry = assemble_dataset(tier_name, tasks, ikmap, ik14_to_db_ids,
                                 ds1_smiles_df, total_events)
        entry["events_dropped_by_exclusion"] = sorted(set(tasks_all) - set(tasks))
        report["tiers"][tier_name] = entry
    json.dump(report, open(os.path.join(OUT, "dataset_ext_build_report.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
