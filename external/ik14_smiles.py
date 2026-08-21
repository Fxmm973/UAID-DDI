# external/ik14_smiles.py
"""IK14 (InChIKey 前14位) -> canonical SMILES via PubChem PUG REST.
InChIKey 前14位是连接层骨架，PUG 的 /inchikey/ 前缀查询对 14 位截断键同样有效；
命中多个 CID 时取第一个并记录。限速 + 断点缓存保证 3170 药量级可稳定跑完。

SMILES 来源契约：PubChem 自 2024-08 数据发布起弃用 CanonicalSMILES，PUG 现返回
ConnectivitySMILES。模块优先读取 CanonicalSMILES，缺失时回退 ConnectivitySMILES，
每条记录写入实际使用的属性名（prop 字段）；map 顶层 __meta__ 记录来源标注。

缓存/复用契约：lookup_smiles_batch 只修改传入的 cache dict，不写盘；断点持久化由
调用方（main）经 on_progress(done, total) 回调执行。内容缺失（no_cid / no_smiles）
与命中结果一样写入缓存；瞬时 HTTP 错误（status_code != 200、网络异常、JSON 解析
失败）重试 2 次后仅标记 failed、不写入缓存，下次运行自动重试。"""
import csv, json, os, sys, time, logging
import requests

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "ik14_smiles_map.json")
META = {"smiles_property": "connectivity_fallback",
        "note": "CanonicalSMILES deprecated by PubChem 2024-08; entries are ConnectivitySMILES"}
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def get_cid(session, ik14):
    r = session.get(f"{BASE}/compound/inchikey/{ik14}/cids/JSON", timeout=30)
    data = r.json()
    if "IdentifierList" in data and data["IdentifierList"]["CID"]:
        return data["IdentifierList"]["CID"][0]
    return None

def get_property(session, cid, prop="CanonicalSMILES,ConnectivitySMILES"):
    # PubChem deprecated CanonicalSMILES (2024-08 data release) -> ConnectivitySMILES.
    # Request both; canonical name takes precedence when present.
    r = session.get(f"{BASE}/compound/cid/{cid}/property/{prop}/JSON", timeout=30)
    prop_list = r.json().get("PropertyTable", {}).get("Properties", [])
    return {str(p["CID"]): p for p in prop_list}

def _lookup_once(session, ik14):
    """单次查询（不重试）：CID 查询 + SMILES 属性查询。瞬时错误以异常上抛。"""
    r = session.get(f"{BASE}/compound/inchikey/{ik14}/cids/JSON", timeout=30)
    if r.status_code != 200:
        raise requests.exceptions.HTTPError(f"cid lookup status {r.status_code}")
    data = r.json()
    if not ("IdentifierList" in data and data["IdentifierList"]["CID"]):
        return {"smiles": None, "cid": None, "prop": None, "status": "failed", "error": "no_cid"}
    cid = data["IdentifierList"]["CID"][0]
    props = get_property(session, cid)
    rec = props.get(str(cid), {})
    smi = rec.get("CanonicalSMILES") or rec.get("ConnectivitySMILES")
    prop = ("CanonicalSMILES" if rec.get("CanonicalSMILES")
            else ("ConnectivitySMILES" if rec.get("ConnectivitySMILES") else None))
    if smi:
        return {"smiles": smi, "cid": cid, "prop": prop, "status": "ok", "error": None}
    return {"smiles": None, "cid": cid, "prop": None, "status": "failed", "error": "no_smiles"}

def _lookup_one(session, ik14, retries=2):
    """带重试的单个查询。瞬时错误（status_code != 200、RequestException、JSON 解析失败）
    按退避 0.5s/1.0s/... 重试 retries 次；耗尽后返回 failed 记录（调用方不写入缓存）。"""
    for attempt in range(retries + 1):
        try:
            return _lookup_once(session, ik14)
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == retries:
                return {"smiles": None, "cid": None, "prop": None, "status": "failed",
                        "error": f"http_error:{type(e).__name__}:{str(e)[:120]}"}
            time.sleep(0.5 * (attempt + 1))

def lookup_smiles_batch(ik14s, cache, on_progress=None):
    """返回 {ik14: {smiles, cid, prop, status, error}}；cache 为持久缓存（含失败项）。

    库函数只修改传入的 cache dict，不写盘；需要断点持久化时，调用方传入
    on_progress(done, total) 回调（每处理完一项调用一次）。内容缺失与命中结果均写入
    cache；瞬时 HTTP 错误（重试 2 次后）仅标记 failed、不写入 cache，下次运行重试。"""
    out, session = {}, requests.Session()
    todo = [k for k in ik14s if k not in cache]
    for i, k in enumerate(sorted(todo)):
        rec = _lookup_one(session, k)
        if rec["status"] == "ok" or rec["error"] in ("no_cid", "no_smiles"):
            cache[k] = rec  # 内容缺失与命中均缓存；瞬时错误不入缓存
        out[k] = rec
        if on_progress is not None:
            on_progress(i + 1, len(todo))
        time.sleep(0.35)  # NCBI 限速：不高于 5 req/s
    for k in ik14s:
        if k not in out:
            out[k] = cache[k]
    return out

def main():
    import pandas as pd
    df = pd_read_pairs()
    ik14s = pd.unique(pd.concat([df["drug_a_ik14"], df["drug_b_ik14"]]))
    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    cache.pop("__meta__", None)  # 来源标注由本函数结尾统一写入

    def checkpoint(done, total):
        if done % 20 == 0:
            json.dump(cache, open(CACHE_PATH, "w"))
            logging.info("checkpoint %d/%d saved", done, total)

    out = lookup_smiles_batch(list(ik14s), cache, on_progress=checkpoint)
    cache["__meta__"] = dict(META)
    json.dump(cache, open(CACHE_PATH, "w"), indent=2)
    with open(os.path.join(os.path.dirname(CACHE_PATH), "mapping_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ik14", "status", "cid", "smiles", "error"])
        for k, v in sorted(out.items()):
            w.writerow([k, v["status"], v["cid"], v["smiles"], v["error"]])
    n_ok = sum(1 for v in out.values() if v["status"] == "ok")
    logging.info("MAPPED %d/%d (%.1f%%)", n_ok, len(out), 100 * n_ok / max(1, len(out)))

def pd_read_pairs():
    import pandas as pd
    # R3: raw data lives in external/raw/ (one dirname, not two)
    raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "ddi_pairs_50k.csv")
    return pd.read_csv(raw, dtype={"drug_a_ik14": "string", "drug_b_ik14": "string",
                                   "faers_best_pt_code_strict": "string"})

if __name__ == "__main__":
    main()
