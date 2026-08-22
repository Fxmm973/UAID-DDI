"""单 split（test2）固定负样本 manifest 生成器。
shared/neg_manifest.py 要求 dev/test/test2 三文件齐全且用 e1rel_e2 排除已知阳性；
新数据集只有 test2 且事件在 DRKG 中无对应边，故此处用任务自身三元组构建排除集合。
R4: tail-corruption 采样池 = 该 tier 全量药物集（tasks 中出现的所有药物并集），
    排除规则 = (a) d_k != d_j，(b) (d_i, d_k) 不在该事件已知阳性 (head, tail) 中，
    (c) 有界：尝试 1000 次后接受任意 d_k != d_j。
"""
import argparse, hashlib, json, os, random
from collections import defaultdict

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]

def tier_drugs_of(tasks):
    """该 tier 全量药物集：tasks 所有三元组中出现过的药物（排序保证确定性）。"""
    return sorted({d for triples in tasks.values() for t in triples for d in (t[0], t[2])})

def generate_manifest_ext(tasks, rel2candidates, seed, no_signal_pairs=None):
    random.seed(seed)
    tier_drugs = tier_drugs_of(tasks)  # R4: 采样池 = tier 全量药物集（rel2candidates 仅作记录）
    manifest = {}
    for event, triples in sorted(tasks.items()):
        known = {(t[0], t[2]) for t in triples}  # (head, tail) 已知阳性
        entries = []
        for d_i, rel, d_j in triples:
            d_k = None
            for _attempt in range(1000):  # R4(c): 有界采样，避免 head 药物与全部药物成阳性时死循环
                cand_k = random.choice(tier_drugs)
                if cand_k != d_j and (d_i, cand_k) not in known:
                    d_k = cand_k
                    break
            if d_k is None:  # R4(c): 1000 次未命中后接受任意 d_k != d_j
                d_k = random.choice([d for d in tier_drugs if d != d_j])
            entries.append([d_i, d_j, d_k, rel])
        manifest[event] = entries
    return manifest

def generate_native_manifest(tasks, no_signal_pairs, seed, few):
    """原生负样本：无信号对池；优先含 d_i 的对。
    为全部三元组（含 support 在内）生成条目，导出端按 [few:] 切片、校验长度。"""
    random.seed(seed)
    by_drug = defaultdict(list)
    for a, b in no_signal_pairs:
        by_drug[a].append(b); by_drug[b].append(a)
    manifest = {}
    for event, triples in sorted(tasks.items()):
        entries = []
        for d_i, rel, d_j in triples:
            pool = by_drug.get(d_i, [])
            d_k = random.choice(pool) if pool else random.choice(no_signal_pairs)[0]
            if d_k == d_i:
                d_k = (pool + [d_j])[0] if pool else d_j
            entries.append([d_i, d_j, d_k, rel])
        manifest[event] = entries
    return manifest

def write_manifest(path, manifest):
    json.dump(manifest, open(path, "w"), indent=2, ensure_ascii=False)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="PharDDIE/dataset_ext_1shot or ..._5shot")
    ap.add_argument("--few", type=int, required=True)
    ap.add_argument("--no-signal-pairs", default=None, help="json of [a,b] pairs (native negatives)")
    args = ap.parse_args()
    tasks = json.load(open(f"{args.dataset}/test2_tasks.json"))
    cand = json.load(open(f"{args.dataset}/rel2candidates.json"))
    no_sig = json.load(open(args.no_signal_pairs)) if args.no_signal_pairs else None
    os.makedirs(f"{args.dataset}/neg_manifests", exist_ok=True)
    hash_log = {}
    for seed in SEEDS:
        m = generate_manifest_ext(tasks, cand, seed)
        h = write_manifest(f"{args.dataset}/neg_manifests/test2_seed{seed}_negatives.json", m)
        hash_log[f"test2_seed{seed}"] = {"path": f"neg_manifests/test2_seed{seed}_negatives.json", "sha256": h}
        if no_sig:
            mn = generate_native_manifest(tasks, no_sig, seed, few=args.few)
            hn = write_manifest(f"{args.dataset}/neg_manifests/test2_seed{seed}_nativenegatives.json", mn)
            hash_log[f"test2_seed{seed}_native"] = {"path": f"neg_manifests/test2_seed{seed}_nativenegatives.json", "sha256": hn}
    json.dump(hash_log, open(f"{args.dataset}/neg_manifests/manifest_hashes.json", "w"), indent=2)
    print("manifests written; hash log saved")

if __name__ == "__main__":
    main()
