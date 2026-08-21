# external/ik14_smiles.py
"""IK14 (InChIKey 前14位) -> canonical SMILES via PubChem PUG REST.
InChIKey 前14位是连接层骨架，PUG 的 /inchikey/ 前缀查询对 14 位截断键同样有效；
命中多个 CID 时取第一个并记录。限速 + 断点缓存保证 3170 药量级可稳定跑完。"""
import csv, json, os, sys, time, logging
import requests

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "ik14_smiles_map.json")
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

def lookup_smiles_batch(ik14s, cache):
    """返回 {ik14: {smiles, cid, status, error}}；cache 为持久缓存（含失败项）。"""
    out, session = {}, requests.Session()
    todo = [k for k in ik14s if k not in cache]
    for i, k in enumerate(sorted(todo)):
        try:
            cid = get_cid(session, k)
            if cid is None:
                out[k] = {"smiles": None, "cid": None, "status": "failed", "error": "no_cid"}
            else:
                props = get_property(session, cid)
                rec = props.get(str(cid), {})
                smi = rec.get("CanonicalSMILES") or rec.get("ConnectivitySMILES")
                out[k] = {"smiles": smi, "cid": cid,
                          "status": "ok" if smi else "failed", "error": None if smi else "no_smiles"}
        except Exception as e:
            out[k] = {"smiles": None, "cid": None, "status": "failed", "error": str(e)[:200]}
        cache[k] = out[k]
        if (i + 1) % 20 == 0:
            json.dump(cache, open(CACHE_PATH, "w"))
            logging.info("checkpoint %d/%d saved", i + 1, len(todo))
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
    out = lookup_smiles_batch(list(ik14s), cache)
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
