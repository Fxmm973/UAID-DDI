#!/usr/bin/env python
# coding=utf-8
"""MedDRA PT code -> term text via UMLS UTS REST API (source MDR).

UMLS MRCONSO 中 source=MEDDRA(MDR) 的 SCUI 即 MedDRA 8 位 PT 码，
/content/current/source/MDR/{code} 返回原子，name 即 PT 术语。
UTS key 通过环境变量 UTS_API_KEY 传入。限速 5 req/s（UTS 上限 20/s）。
输出：outputs/pt_labels_uts.json {code: {label, ui, n}}。
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://uts-ws.nlm.nih.gov/rest"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def fetch(key, url):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {key}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None

def main():
    key = os.environ.get("UTS_API_KEY")
    if not key:
        sys.exit("UTS_API_KEY env not set")
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(OUT), "raw", "ddi_pairs_50k.csv"),
                     dtype={"faers_best_pt_code_strict": "string"})
    codes = sorted(df[df["faers_ror95_lcl_max_strict"].notna()]
                   ["faers_best_pt_code_strict"].dropna().unique())
    out_path = os.path.join(OUT, "pt_labels_uts.json")
    out = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for i, code in enumerate(codes):
        if code in out and out[code].get("label"):
            continue
        d = fetch(key, f"{BASE}/content/current/source/MDR/{code}")
        label, ui = None, None
        if d and d.get("result"):
            label = d["result"].get("name")
            ui = d["result"].get("ui")
        out[code] = {"label": label or f"FAERS adverse event PT-{code}", "ui": ui}
        if (i + 1) % 25 == 0:
            n_hit = sum(1 for v in out.values() if v.get("ui"))
            print(f"progress {i+1}/{len(codes)}; hits: {n_hit}", flush=True)
            json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
        time.sleep(0.2)
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    n_hit = sum(1 for v in out.values() if v.get("ui"))
    print(f"DONE. uts labels: {n_hit}/{len(codes)} ({100 * n_hit / max(1, len(codes)):.1f}%)", flush=True)

if __name__ == "__main__":
    main()
