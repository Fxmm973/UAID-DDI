#!/usr/bin/env python
# coding=utf-8
"""MedDRA PT code -> term label extraction from PubMed literature.

RxPairEvid redistributes MedDRA PT *codes* only (term text excluded for
MedDRA licensing). This script recovers English PT labels from published
papers that cite the code, e.g. "Torsade de pointes (10020772)" or
"MedDRA PT 10020772: Torsade de pointes". A label is kept only when the
same normalized phrase is found in >= MIN_EVIDENCE distinct mentions;
codes without sufficient support fall back to "FAERS adverse event PT-{code}".
Output: outputs/pt_labels.json {code: {label, source, n_evidence, pmids}}.
Rate limit: Entrez without API key = 3 req/s; esearch per code, efetch batched.
"""
import json, os, re, sys, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from collections import Counter

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MIN_EVIDENCE = 2

def get(url, **params):
    params = dict(params, retmode="json", tool="UAID-DDI-external", email="uaid.ddi.validation@gmail.com")
    req = urllib.request.Request(EUTILS + url + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "UAID-DDI-external/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return ""

def esearch_pmids(code):
    txt = get("/esearch.fcgi", db="pubmed", term=f'"{code}"[All Fields]', retmax=5)
    try:
        return json.loads(txt)["esearchresult"]["idlist"]
    except Exception:
        return []

def efetch_abstracts(pmids):
    """Batch efetch abstracts; returns {pmid: text}."""
    out = {}
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i + 200]
        txt = get("/efetch.fcgi", db="pubmed", id=",".join(chunk), rettype="abstract")
        try:
            root = ET.fromstring(txt)
            for art in root.findall(".//PubmedArticle"):
                pmid = art.findtext(".//PMID") or ""
                parts = [t.text or "" for t in art.findall(".//AbstractText")]
                out[pmid] = " ".join(parts)
        except Exception:
            pass
        time.sleep(0.4)
    return out

PHRASE = r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,6}"
STOPWORDS = {"patient", "patients", "study", "the", "a", "an", "of", "in", "and", "was", "were",
             "adverse", "event", "events", "term", "terms", "meddra", "pt", "reported", "case",
             "rate", "group", "data", "table", "using", "with", "for", "from", "medication"}

def extract_phrases(text, code):
    """Return candidate label phrases for THIS code in one abstract.
    Patterns bind the exact code (never a different 8-digit number):
    "Torsade de pointes (10020772)" / "10020772: Torsade de pointes" /
    "Torsade de pointes [10020772]"."""
    pats = [
        re.compile(rf"({PHRASE})\s*[\(\[]\s*{code}\s*[\)\]]", re.I),
        re.compile(rf"{code}\s*[\)\]]?\s*[:=(]+\s*({PHRASE})", re.I),
        re.compile(rf"({PHRASE})\s*\[?\s*{code}\b", re.I),
    ]
    cands = []
    for pat in pats:
        for m in pat.finditer(text):
            for phrase in m.groups():
                if not phrase:
                    continue
                phrase = re.sub(r"\s+", " ", phrase).strip(" ,;:.-()[]")
                words = [w for w in phrase.split() if w.lower() not in STOPWORDS and w.lower() != "pt"]
                if 1 <= len(words) <= 7 and phrase[:1].isupper():
                    cands.append(" ".join(words))
    return cands

def load_codes():
    import pandas as pd
    raw = os.path.join(os.path.dirname(OUT_DIR), "raw", "ddi_pairs_50k.csv")
    df = pd.read_csv(raw, dtype={"faers_best_pt_code_strict": "string"})
    codes = sorted(df[df["faers_ror95_lcl_max_strict"].notna()]
                   ["faers_best_pt_code_strict"].dropna().unique())
    return list(codes)

def main():
    codes = load_codes()
    out = {}
    for i, code in enumerate(codes):
        pmids = esearch_pmids(code)
        if pmids:
            texts = efetch_abstracts(pmids)
            votes = Counter()
            pmid_ev = {}
            for pmid, txt in texts.items():
                for phrase in extract_phrases(txt, code):
                    votes[phrase] += 1
                    pmid_ev.setdefault(phrase, []).append(pmid)
            best = votes.most_common(1)
            if best and best[0][1] >= MIN_EVIDENCE:
                out[code] = {"label": best[0][0], "source": "pubmed",
                             "n_evidence": best[0][1], "pmids": pmid_ev[best[0][0]][:3]}
            else:
                out[code] = {"label": f"FAERS adverse event PT-{code}", "source": "fallback",
                             "n_evidence": 0, "pmids": []}
        else:
            out[code] = {"label": f"FAERS adverse event PT-{code}", "source": "fallback",
                         "n_evidence": 0, "pmids": []}
        if (i + 1) % 25 == 0:
            n_hit = sum(1 for v in out.values() if v["source"] == "pubmed")
            print(f"progress {i + 1}/{len(codes)}; pubmed hits so far: {n_hit}", flush=True)
            json.dump(out, open(os.path.join(OUT_DIR, "pt_labels.json"), "w"), indent=2, ensure_ascii=False)
        time.sleep(0.34)  # Entrez: <=3 req/s without key
    json.dump(out, open(os.path.join(OUT_DIR, "pt_labels.json"), "w"), indent=2, ensure_ascii=False)
    n_hit = sum(1 for v in out.values() if v["source"] == "pubmed")
    print(f"DONE. pubmed labels: {n_hit}/{len(codes)} ({100 * n_hit / max(1, len(codes)):.1f}%)", flush=True)

if __name__ == "__main__":
    main()
