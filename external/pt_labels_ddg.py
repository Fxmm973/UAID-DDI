#!/usr/bin/env python
# coding=utf-8
"""MedDRA PT code -> label mining from DuckDuckGo result snippets.

MedDRA term text is license-restricted; PT codes however appear in countless
public pharmacovigilance pages, PDFs and trial registries indexed by search
engines, co-occurring with their labels (e.g. "Hypertension (10020772)").
For each code we fetch DDG HTML results, extract candidate label phrases
from the snippets, and accept a label only when >= MIN_VOTES distinct
snippets agree. Unresolved codes fall back to "FAERS adverse event PT-{code}".
Outputs: outputs/pt_labels_ddg.json (+ checkpoint every 20 codes).
"""
import html as htmlmod
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MIN_VOTES = 2
PHRASE = r"[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,6}"
STOP = {"meddra", "pt", "term", "code", "adverse", "event", "events", "reaction",
        "reactions", "patient", "patients", "study", "table", "report", "reported"}

def fetch_snippets(code):
    q = f'"{code}" meddra'
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            page = r.read().decode("utf-8", "replace")
    except Exception:
        return []
    snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    texts = []
    for s in snips:
        t = re.sub(r"<[^>]+>", " ", s)
        t = htmlmod.unescape(t)
        texts.append(re.sub(r"\s+", " ", t))
    return texts

def candidates(text, code):
    out = []
    for m in re.finditer(rf"({PHRASE})\s*[\(\[]?\s*{code}", text):
        out.append(m.group(1).strip())
    for m in re.finditer(rf"{code}\s*[\)\]]?\s*[:=(]?\s*({PHRASE})", text):
        out.append(m.group(1).strip())
    return out

def main():
    import pandas as pd
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "raw", "ddi_pairs_50k.csv"),
                     dtype={"faers_best_pt_code_strict": "string"})
    codes = sorted(df[df["faers_ror95_lcl_max_strict"].notna()]
                   ["faers_best_pt_code_strict"].dropna().unique())
    out_path = os.path.join(OUT, "pt_labels_ddg.json")
    out = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for i, code in enumerate(codes):
        if code in out:
            continue
        votes = Counter()
        for text in fetch_snippets(code):
            for cand in candidates(text, code):
                words = [w for w in cand.split() if w.lower() not in STOP]
                if words and words[0][0].isupper():
                    votes[" ".join(words)] += 1
        best = votes.most_common(1)
        out[code] = {"label": best[0][0], "n": best[0][1]} if best and best[0][1] >= MIN_VOTES \
            else {"label": f"FAERS adverse event PT-{code}", "n": 0}
        if (i + 1) % 20 == 0:
            n_hit = sum(1 for v in out.values() if v.get("n", 0) >= MIN_VOTES)
            print(f"progress {i+1}/{len(codes)}; hits: {n_hit}", flush=True)
            json.dump(out, open(out_path, "w"), indent=2)
        time.sleep(1.5)
    json.dump(out, open(out_path, "w"), indent=2)
    n_hit = sum(1 for v in out.values() if v.get("n", 0) >= MIN_VOTES)
    print(f"DONE. ddg labels: {n_hit}/{len(codes)} ({100 * n_hit / max(1, len(codes)):.1f}%)", flush=True)

if __name__ == "__main__":
    main()
