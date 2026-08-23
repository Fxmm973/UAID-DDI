#!/usr/bin/env python
# coding=utf-8
"""Task 17: case-evidence upgrade for the dataset-2 case study —
Europe PMC full-text search (leg A) + class-level mechanism search (leg B).

Background: the R29 three-tier evidence produced by case_study_per_event.py
(PubMed abstracts only, pair co-occurrence queries) left 21/25 candidates as
"none" — class-level pharmacological mechanisms (erythromycin QTc
prolongation, mineralocorticoid + thiazide hypokalemia, beta-agonist vs
neuromuscular blockade, ...) were never searched. This script is the final
evidence-layer upgrade.

Two retrieval legs, fully automated and reproducible (every query string is
recorded in the markdown output; every HTTP response is cached in
external/outputs/case_evidence_upgrade_cache.json so re-runs are offline and
byte-stable):

Leg A — direct evidence upgrade (Europe PMC full text):
  q1 = ("<name_a>" AND "<name_b>") AND (interaction OR adverse OR effect)
       pageSize=5, resultType=core (pmid, title, inEPMC/inPMC full-text flags)
  q2 = ("<name_a>" AND "<name_b>")  pageSize=3  (wider co-occurrence probe)
  STRICT direct tier (controller ruling, 2026-08-23): both drug names must
  co-occur in ONE SENTENCE of the title/abstract together with an
  interaction/adverse keyword — the pair must be SIMULTANEOUSLY discussed,
  verifiable from the title/abstract. Full-text co-occurrence is only a
  probe (reported, never drives the tier); drug lists / docking contexts
  fail. Every direct citation is additionally re-verified by re-fetching its
  REAL title via NCBI eutils esummary (title must match the claim).

Leg B — class-level mechanism evidence (per event direction):
  For EACH drug separately:  "<drug>" AND (<mech keywords>), pageSize=10,
  keywords chosen from the event text per the category table in the task
  spec. A side counts as supported when an abstract contains the drug name
  AND a role keyword consistent with the event direction (hand-curated
  per-candidate role table below — the claimed role must literally appear in
  the retrieved literature) AND the record's REAL esummary title is
  consistent with the claim (title contains the drug name or a role/mech
  keyword). Tier = class_suggested when >=1 side is supported and
  direction-consistent. Conservative rule: prefer "none" over a forced
  label (宁缺毋滥). v1 tiers are NOT carried over — they are re-derived from
  scratch (v1 citations fail the strict standard).

Tier precedence: direct > class_suggested > none.

Outputs (new files only; existing files untouched):
  external/outputs/case_evidence_dataset2_v2.md
  external/outputs/case_candidates_dataset2_per_event_v2.csv
  external/outputs/case_evidence_upgrade_cache.json (HTTP cache, reproducible)
"""
import argparse
import datetime
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(OUT, "outputs")
CAND_CSV = os.path.join(OUTDIR, "case_candidates_dataset2_per_event.csv")
V2_MD = os.path.join(OUTDIR, "case_evidence_dataset2_v2.md")
V2_CSV = os.path.join(OUTDIR, "case_candidates_dataset2_per_event_v2.csv")
CACHE_JSON = os.path.join(OUTDIR, "case_evidence_upgrade_cache.json")

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
EUTILS_SUMMARY = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                  "?db=pubmed&retmode=json&tool=PharDDIE-ext-validation&id={pmid}")
UA = {"User-Agent": "PharDDIE-ext-validation/1.0 (case evidence upgrade)"}
RATE_LIMIT_S = 0.5          # >= 0.4 s/request (politeness rule)
ATTEMPTS = 3
BACKOFF_S = 2.0
PAGE_A = 5                  # leg A q1 top-5
PAGE_A_WIDE = 3             # leg A q2 top-3
PAGE_B_SCAN = 10            # leg B per-drug scan depth (headline list = top-3)
WINDOW_CHARS = 150
NO_PREPRINT = True          # PPR (preprint) records never drive evidence tiers

# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

# DB ids with no name in the candidates CSV, resolved from go.drugbank.com
# (queried 2026-08-23). Used for both search legs; the CSV a_name column is
# left untouched (controller ruling: v1 columns are preserved).
NAME_OVERRIDE = {"DB14006": "Choline salicylate"}


def query_term(name):
    """Sanitize a drug name into a safe EPMC phrase term: strip stereo
    prefixes ((S)-, (R)-, (+)-, ...) and characters that Europe PMC's query
    parser treats as operators (parens, '+', ...). The original name is
    always recorded in the markdown output."""
    if name is None or (isinstance(name, float) and name != name):
        name = ""  # pandas NaN
    cleaned = str(name)
    cleaned = re.sub(r"^\([SR\+\-]+\)[- ]*", "", cleaned)  # stereo prefix
    cleaned = re.sub(r"[^A-Za-z0-9 '\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else ""


# ---------------------------------------------------------------------------
# HTTP: Europe PMC with cache + rate limit + retry
# ---------------------------------------------------------------------------

_CACHE = None


def _load_cache():
    global _CACHE
    if _CACHE is None:
        if os.path.exists(CACHE_JSON):
            with open(CACHE_JSON, encoding="utf-8") as f:
                _CACHE = json.load(f)
        else:
            _CACHE = {}
    return _CACHE


def _save_cache():
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(_CACHE, f, ensure_ascii=False, indent=0, sort_keys=True)


def _http_get(url):
    cache = _load_cache()
    if url in cache:
        return cache[url]
    last = None
    for i in range(ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = resp.read().decode("utf-8")
            cache[url] = body
            _save_cache()
            return body
        except Exception as exc:  # network / HTTP errors
            last = exc
            if i < ATTEMPTS - 1:
                time.sleep(BACKOFF_S * (i + 1))
    raise RuntimeError(f"EPMC request failed after {ATTEMPTS} attempts: {last}\n{url}")


def epmc_search(query, page_size, sort=None):
    """Europe PMC REST search; returns the result-list entries (core fields:
    id/source/pmid/pmcid/title/abstractText/inEPMC/inPMC). With sort=None the
    API default applies (relevance ordering) — it surfaces the mechanism and
    co-occurrence literature that date sorting buries (e.g. erythromycin QTc
    classics). NB: Europe PMC REJECTS an explicit sort=relevance parameter,
    so the default is used and recorded as "relevance (default)". Results are
    frozen in the HTTP cache, so re-runs are byte-stable."""
    params = {"query": query, "format": "json", "pageSize": page_size,
              "resultType": "core"}
    if sort:
        params["sort"] = sort
    url = EPMC_SEARCH + "?" + urllib.parse.urlencode(params)
    time.sleep(RATE_LIMIT_S)
    return json.loads(_http_get(url)).get("resultList", {}).get("result", [])


def epmc_fulltext_xml(pmcid):
    """Europe PMC full-text XML for one PMC article (leg A full-text check)."""
    url = EPMC_FULLTEXT.format(pmcid=urllib.parse.quote(pmcid))
    time.sleep(RATE_LIMIT_S)
    return _http_get(url)


def strip_tags(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fulltext_body_text(xml):
    """Extract the meaningful text of a Europe PMC fullTextXML response:
    abstract + body only (front matter / reference lists are dropped, they
    caused false 'interact' hits)."""
    parts = []
    for tag in ("abstract", "body"):
        m = re.search(r"<%s>.*?</%s>" % (tag, tag), xml, re.S)
        if m:
            parts.append(strip_tags(m.group(0)))
    return " ".join(p for p in parts if p).strip()


def hit_pmid(hit):
    return hit.get("pmid") or (hit.get("id") if hit.get("source") == "MED" else "")


# ---------------------------------------------------------------------------
# Hard PMID verification (controller ruling, 2026-08-23): every cited PMID is
# re-fetched from NCBI eutils esummary and its REAL title recorded; citations
# whose title is inconsistent with the evidence claim are deleted.
# ---------------------------------------------------------------------------


def esummary_title(pmid):
    """Real title for a PubMed PMID via NCBI eutils esummary (JSON), cached.
    Returns (title, ok) where ok=False means the record could not be
    confirmed (no PMID / network failure)."""
    pmid = str(pmid or "").strip()
    if not pmid or not pmid.isdigit():
        return "", False
    url = EUTILS_SUMMARY.format(pmid=pmid)
    time.sleep(RATE_LIMIT_S)
    try:
        data = json.loads(_http_get(url))
    except (RuntimeError, ValueError):
        return "", False
    rec = (data.get("result") or {}).get(pmid) or {}
    title = (rec.get("title") or "").strip()
    return title, bool(title)


def _stem(name):
    """Case-insensitive substring stem for loose title matching: the name
    itself or its last-char-trimmed form (catches plurals and inflections
    like 'Aldosterone' -> 'Hyperaldosteronism')."""
    n = (name or "").lower().strip()
    return [n] if len(n) < 5 else [n, n[:-1]]


def title_consistent(real_title, names, kws):
    """Does the REAL title of the cited paper match the evidence claim?
    True when the title contains one of the drug names (or stems) or one of
    the claim's role/mechanism keywords. A title containing none of them
    contradicts the claim -> the citation must be deleted (宁缺毋滥)."""
    t = (real_title or "").lower()
    if not t:
        return False
    for nm in names:
        if any(s in t for s in _stem(nm)):
            return True
    return any(k.lower() in t for k in (kws or []) if k)


def verify_record(rec, names, kws):
    """Add the REAL esummary title to an evidence record and check
    consistency; rec becomes unusable (ok=False) when the title cannot be
    confirmed or contradicts the claim."""
    title, ok = esummary_title(rec.get("pmid") or rec.get("id"))
    if not ok and rec.get("title"):
        # no PMID (PMC-only record): the EPMC core title is the real title
        title, ok = rec["title"], True
    rec["real_title"] = title
    rec["title_ok"] = ok and title_consistent(title, names, kws)
    return rec


# ---------------------------------------------------------------------------
# Direct tier (leg A): same conservative rule as case_study_per_event (R29).
# CONTEXT_KW mirrors case_study_ext; DIRECT_KW = CONTEXT_KW + effect variants.
# ---------------------------------------------------------------------------

CONTEXT_KW = ("interact", "advers", "drug-drug", "ddi", "combination",
              "concomitant", "toxicity", "overdose", "side effect")
DIRECT_KW = CONTEXT_KW + ("additive", "synergis", "potentiat")

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _name_re(name):
    """Case-insensitive word-boundary matcher for a drug name, tolerant of a
    simple plural suffix (identical to case_study_per_event._name_re)."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) +
                      r"(?:s|es)?(?![A-Za-z0-9])", re.IGNORECASE)


def direct_check(text, a_term, b_term, same_sentence_only=True):
    """STRICT direct rule (controller ruling, 2026-08-23): both drug names
    must co-occur in ONE SENTENCE of the title/abstract, and the sentence
    must contain an interaction/adverse keyword. Verifiable from the
    title/abstract alone — full text no longer drives the direct tier (it
    produced drug-list false positives). Returns (fragment, matched_kw)."""
    if not a_term or not b_term:
        return None
    ra, rb = _name_re(a_term), _name_re(b_term)
    norm = strip_tags(text)
    if not (ra.search(norm) and rb.search(norm)):
        return None
    for sent in _SENT_RE.split(norm):
        if ra.search(sent) and rb.search(sent):
            low = sent.lower()
            kw = next((k for k in DIRECT_KW if k in low), None)
            if kw:
                return sent, kw
    return None


# ---------------------------------------------------------------------------
# Class-level (leg B): mechanism keywords per event + per-drug role table.
# ---------------------------------------------------------------------------
# ROLES[event] = {
#   "mech":    query keywords for the event direction (task vocabulary),
#   "a"/"b":   {kws: role keywords that must literally appear in an abstract
#              alongside the drug name for the side to count as supported,
#              ok: is that role consistent with the event direction?,
#              note: the claimed mechanism (one sentence)},
#   "summary": one-sentence mechanism explanation (evidence_note column).
# }
# The claimed roles are standard pharmacology; the script VERIFIES each one
# against the retrieved abstracts — a side only counts when the literature
# actually contains the role keywords together with the drug name.

ROLES = {
    "The risk or severity of hypertension decrease": {
        "mech": ["hypotens", "vasodilat", "vasorelax", "calcium channel",
                 "pressor", "alpha-adrenergic"],
        "a": {"kws": ["hypotens", "vasodilat", "vasorelax", "calcium channel"],
              "ok": True,
              "note": "Dexniguldipine 为二氢吡啶类钙通道阻滞剂，具血管舒张/降压作用"},
        "b": {"kws": ["pressor", "vasoconstrict", "alpha-adrenergic"],
              "ok": True,
              "note": "Phenylephrine 为 α1 激动剂升压药，其升压作用可被拮抗"},
        "summary": "Dexniguldipine（钙通道阻滞剂）的降压作用与 Phenylephrine（α1 升压药）的拮抗，增加血压降低风险。"},
    "The absorption decrease": {
        "mech": ["absorption", "bioavailability", "transporter"],
        "a": {"kws": ["absorption", "gastric emptying", "motility",
                      "anticholinergic", "bioavailab"],
              "ok": True,
              "note": "左旋多巴在小肠吸收，胃肠动力/排空改变可减少其吸收"},
        "b": {"kws": ["absorption", "motility", "gastric emptying",
                      "antimuscarinic", "opioid"],
              "ok": True,
              "note": "Trimebutine 调节胃肠动力（阿片受体激动/弱抗胆碱），可改变左旋多巴吸收"},
        "summary": "Trimebutine 的胃肠动力调节作用可改变左旋多巴经小肠的吸收，增加吸收减少风险。"},
    "The serum concentration of the active metabolites increase": {
        "mech": ["metabolite", "CYP", "plasma concentration"],
        "a": {"kws": ["active metabolite", "cyp", "plasma concentration",
                      "serum concentration", "2-hydroxydesipramine"],
              "ok": True,
              "note": "Desipramine 既是活性代谢物（imipramine 经 CYP 代谢生成）也是 CYP2D6 底物，其血浆浓度受 CYP2D6 活性影响"},
        "b": {"kws": ["active metabolite", "cyp", "plasma concentration",
                      "serum concentration", "2-hydroxydesipramine"],
              "ok": True,
              "note": "同一药物两侧：Desipramine 同时承担母药与活性代谢物角色，血浆浓度均受 CYP2D6 活性影响"},
        "summary": "Desipramine 为 CYP2D6 底物与活性代谢物角色；CYP2D6 活性降低可提高其（及母药）血浆浓度。"},
    "the neuromuscular blocking activities decrease": {
        "mech": ["neuromuscular block", "muscle relaxant"],
        "a": {"kws": ["neuromuscular block", "muscle relaxant", "beta-adrenergic",
                      "beta-2", "catecholamine"],
              "ok": True,
              "note": "β2 激动剂（Terbutaline）与神经肌肉阻滞药存在药效学相互作用，可减弱非去极化肌松作用"},
        "b": {"kws": ["neuromuscular block", "muscle relaxant", "pancuronium",
                      "vecuronium", "curare"],
              "ok": True,
              "note": "Pipecuronium 为非去极化神经肌肉阻滞药，其阻滞效应可被拟交感药物减弱"},
        "summary": "Terbutaline（β2 激动剂）与 Pipecuronium（非去极化肌松药）合用时，β 肾上腺素能激活可减弱神经肌肉阻滞效应。"},
    "The risk or severity of renal failure hypotension and hyperkalemia increase": {
        "mech": ["excretion", "renal", "hypotens", "hypokalemia", "potassium"],
        "a": {"kws": ["hypotens", "volume deplet", "dehydrat", "diuretic"],
              "ok": True,
              "note": "SGLT2 抑制剂（Ertugliflozin）渗透性利尿→容量不足/低血压，加重肾脏低灌注"},
        "b": {"kws": ["hyperkalemi", "hypotens", "renal fail", "renal impairment",
                      "angiotensin", "ace inhibitor"],
              "ok": True,
              "note": "ACE 抑制剂（Quinapril）可致高钾血症、低血压与肾功能损害"},
        "summary": "Ertugliflozin（SGLT2 抑制剂）的利尿/降压作用与 Quinapril（ACE 抑制剂）的高钾、低血压与肾功能影响叠加，增加肾衰/低血压/高钾血症风险。"},
    "The risk or severity of renal failure increase": {
        "mech": ["excretion", "renal"],
        "a": {"kws": ["renal fail", "acute kidney injury", "kidney injury",
                      "prostaglandin", "nsaid", "salicylate", "nephrotox"],
              "ok": True,
              "note": "Choline salicylate（NSAID）抑制肾脏前列腺素合成→肾灌注下降/肾功能损害"},
        "b": {"kws": ["hyperkalemi", "acute kidney injury", "renal fail",
                      "renal impairment", "angiotensin", "ace inhibitor"],
              "ok": True,
              "note": "Ramipril（ACE 抑制剂）改变肾血流动力学，低灌注下加重肾功能损害"},
        "summary": "Choline salicylate（NSAID）与 Ramipril（ACEI）合用：NSAID 抑制肾前列腺素合成叠加 ACEI 的肾血流动力学效应，增加肾衰竭风险（双重打击）。"},
    "The risk or severity of QTc prolongation and hypotension increase": {
        "mech": ["QTc", "QT prolongation", "torsades", "hypotens", "vasodilat"],
        "a": {"kws": ["qtc", "qt prolongation", "torsades", "herg", "qt interval"],
              "ok": True,
              "note": "Erythromycin 为已知 QTc 延长/尖端扭转型室速药物（hERG 阻滞）"},
        "b": {"kws": ["qtc", "qt interval", "herg", "hypotens", "calcium channel"],
              "ok": True,
              "note": "Dexniguldipine 为二氢吡啶钙通道阻滞剂（降压）并具 hERG 阻滞作用"},
        "summary": "Erythromycin（hERG 阻滞，QTc 延长）与 Dexniguldipine（钙通道阻滞/降压，hERG 阻滞）叠加，增加 QTc 延长与低血压风险。"},
    "the stimulatory activities decrease": {
        "mech": ["stimulant", "dopamine"],
        "a": {"kws": ["dopamine", "d2 receptor", "d2 antagonist"],
              "ok": True,
              "note": "Lumateperone 为抗精神病药，同时调节 5-HT/多巴胺/谷氨酸神经传递（含多巴胺能机制）"},
        "b": {"kws": ["stimulant", "amphetamine", "dopamine", "psychostimulant"],
              "ok": True,
              "note": "Lisdexamfetamine 为苯丙胺类中枢兴奋剂，具拟交感刺激活性"},
        "summary": "Lumateperone 的多巴胺能调节作用可减弱 Lisdexamfetamine（苯丙胺类兴奋剂）的中枢刺激活性。"},
    "The risk or severity of angioedema increase": {
        "mech": ["angioedema"],
        "a": {"kws": ["angioedema"],
              "ok": True,
              "note": "Sitagliptin（DPP-4 抑制剂）与血管性水肿相关"},
        "b": {"kws": ["angioedema"],
              "ok": True,
              "note": "Alogliptin（DPP-4 抑制剂）同类——同类相加"},
        "summary": "Sitagliptin 与 Alogliptin 同为 DPP-4 抑制剂，同类叠加增加血管性水肿风险。"},
    "The risk or severity of myopathy and weakness increase": {
        "mech": ["myopathy", "rhabdomyolysis"],
        "a": {"kws": ["myopathy", "rhabdomyolysis", "weakness"],
              "ok": True,
              "note": "Pyrantel 未见明确肌病机制（文献支持需人工核实）"},
        "b": {"kws": ["myopathy", "rhabdomyolysis", "steroid", "corticosteroid",
                      "muscle weakness"],
              "ok": True,
              "note": "糖皮质激素（Budesonide）可致类固醇肌病"},
        "summary": "糖皮质激素（Budesonide）的类固醇肌病作用为主要机制，增加肌病/肌无力风险。"},
    "The risk or severity of QTc prolongation decrease": {
        "mech": ["QTc", "QT prolongation", "torsades"],
        "a": {"kws": ["qtc", "qt prolongation", "torsades", "qt interval"],
              "ok": False,  # mefloquine PROLONGS QT -> opposite of the event
              "note": "(+)-Mefloquine 本身延长 QT——与事件方向相反，不计"},
        "b": {"kws": ["qt interval", "qtc", "beta-block", "beta adrenergic",
                      "propranolol", "shorten"],
              "ok": True,
              "note": "Befunolol 为 β 受体阻滞剂，β 阻滞可缩短/抑制 QT 延长"},
        "summary": "Befunolol 的 β 受体阻滞作用可缩短或抑制 QT 间期延长，从而降低 Mefloquine 所致 QTc 延长风险。"},
    "the thrombogenic activities increase": {
        "mech": ["thrombosis", "coagulation"],
        "a": {"kws": ["thrombogen"],
              "ok": True,
              "note": "Niflumic acid 未见促血栓机制文献（需人工核实）"},
        "b": {"kws": ["thrombosis", "coagulation", "estrogen", "contraceptive",
                      "venous thromboembolism"],
              "ok": True,
              "note": "Mestranol（雌激素，炔雌醇前药）为已知促血栓因素（口服避孕药相关血栓）"},
        "summary": "Mestranol 的雌激素促血栓作用（凝血因子改变）增加血栓形成风险。"},
    "The risk or severity of neutropenia and thrombocytopenia increase": {
        "mech": ["neutropenia", "myelosuppression"],
        "a": {"kws": ["neutropeni", "myelosuppress", "thrombocytopeni"],
              "ok": True,
              "note": "噻嗪类利尿剂罕见血细胞减少"},
        "b": {"kws": ["neutropeni", "myelosuppress", "thrombocytopeni"],
              "ok": True,
              "note": "HDAC 抑制剂（Belinostat）骨髓抑制——中性粒细胞/血小板减少为已知毒性"},
        "summary": "Belinostat（HDAC 抑制剂）的骨髓抑制毒性（中性粒细胞/血小板减少）为主要机制。"},
    "The risk or severity of sedation and somnolence increase": {
        "mech": ["sedation", "CNS depressant", "GABA"],
        "a": {"kws": ["sedat", "antihistamin", "tricyclic", "cns depress",
                      "somnolence"],
              "ok": True,
              "note": "Cidoxepin 为三环类抗抑郁药（组胺 H1 拮抗→镇静）"},
        "b": {"kws": ["sedat", "somnolence", "dopamine deplet", "vmat",
                      "cns depress"],
              "ok": True,
              "note": "Deutetrabenazine 为 VMAT2 抑制剂，嗜睡/镇静为已知不良反应"},
        "summary": "Cidoxepin（三环类，H1 拮抗）与 Deutetrabenazine（VMAT2 抑制）均有中枢镇静作用，叠加增加镇静/嗜睡风险。"},
    "The risk or severity of Tachycardia and drowsiness increase": {
        "mech": ["sedation", "CNS depressant", "GABA"],
        "a": {"kws": [],
              "ok": False,
              "note": "Trimebutine 未见嗜睡/心动过速主要机制文献"},
        "b": {"kws": ["drowsiness", "sedat", "somnolence", "tachycard",
                      "cannabinoid"],
              "ok": True,
              "note": "Nabilone 为合成大麻素受体激动剂，嗜睡为已知不良反应"},
        "summary": "Nabilone（合成大麻素）的中枢作用（嗜睡、心动过速）为主要机制。"},
    "the vasopressor activities increase": {
        "mech": ["pressor", "vasopressor", "vasoconstrict"],
        "a": {"kws": ["pressor", "vasoconstrict", "ergot", "oxytocic",
                      "blood pressure", "5-hydroxytryptamine"],
              "ok": True,
              "note": "Ergometrine 为麦角碱类宫缩剂，强效血管收缩/升压作用"},
        "b": {"kws": [],
              "ok": False,
              "note": "Tianeptine 未见升压主要机制文献"},
        "summary": "Ergometrine 的血管收缩/升压作用（麦角碱类）为主要机制。"},
    "The risk or severity of neutropenia increase": {
        "mech": ["neutropenia", "myelosuppression"],
        "a": {"kws": ["neutropeni", "myelosuppress", "agranulocytosis"],
              "ok": True,
              "note": "Amitriptyline 罕见粒细胞缺乏"},
        "b": {"kws": ["neutropeni", "myelosuppress", "topoisomerase"],
              "ok": True,
              "note": "Irinotecan 为化疗药，中性粒细胞减少为常见剂量限制性毒性"},
        "summary": "Irinotecan 的骨髓抑制（中性粒细胞减少）为主要机制。"},
    "an increase in the absorption resulting in an increased serum concentration and potentially a worsening of adverse effects cause": {
        "mech": ["absorption", "bioavailability", "transporter"],
        "a": {"kws": ["absorption", "gastric emptying", "motility",
                      "antimuscarinic", "anticholinergic", "bioavailab"],
              "ok": True,
              "note": "Methantheline 为强效抗毒蕈碱药（抗胆碱作用较阿托品更强更持久）"},
        "b": {"kws": ["absorption", "bioavailab", "plasma concentration",
                      "serum concentration", "transporter"],
              "ok": True,
              "note": "Raltegravir 的吸收/血浆浓度对伴随药物（如抗酸药）敏感"},
        "summary": "Methantheline 的抗胆碱作用（减慢胃肠排空）可增加 Raltegravir 的吸收，提高其血清浓度。"},
    "The risk or severity of fluid retention increase": {
        "mech": ["fluid retention", "edema"],
        "a": {"kws": ["fluid retention", "edema", "mineralocorticoid", "sodium",
                      "water retention"],
              "ok": True,
              "note": "Aldosterone 为盐皮质激素，促钠水潴留"},
        "b": {"kws": ["fluid retention", "edema"],
              "ok": True,
              "note": "Danazol 为合成雄激素，可致液体潴留/水肿"},
        "summary": "盐皮质激素（Aldosterone）与合成雄激素（Danazol）均促钠水潴留，叠加增加液体潴留风险。"},
    "the hypokalemic activities increase": {
        "mech": ["hypokalemia", "potassium", "electrolyte"],
        "a": {"kws": ["hypokalemi", "potassium", "mineralocorticoid"],
              "ok": True,
              "note": "Aldosterone 促钾排泄→低钾血症"},
        "b": {"kws": ["hypokalemi", "potassium", "diuretic", "thiazide"],
              "ok": True,
              "note": "噻嗪样利尿剂（Indapamide）低钾血症为已知不良反应"},
        "summary": "盐皮质激素（Aldosterone）与噻嗪样利尿剂（Indapamide）均促钾丢失，叠加增加低钾血症风险。"},
    "The risk or severity of renal failure and hypertension increase": {
        "mech": ["excretion", "renal", "hypotens", "blood pressure"],
        "a": {"kws": ["renal fail", "acute kidney injury", "kidney injury",
                      "nephrotox", "prostaglandin", "nsaid"],
              "ok": True,
              "note": "NSAID（Dexketoprofen）抑制肾前列腺素→肾功能损害"},
        "b": {"kws": ["nephrotoxic", "renal fail", "acute kidney injury",
                      "hypertension", "blood pressure", "calcineurin"],
              "ok": True,
              "note": "Ciclosporin 具肾毒性并升高血压；与 NSAID 肾毒性叠加"},
        "summary": "NSAID（Dexketoprofen）与 Ciclosporin 均具肾毒性，且 Ciclosporin 致高血压——肾衰与高血压风险叠加。"},
    "The risk or severity of hyponatremia increase": {
        "mech": ["hyponatremia", "sodium"],
        "a": {"kws": ["hyponatremi", "siadh", "sodium"],
              "ok": True,
              "note": "SSRI（Zimelidine）可致 SIADH/低钠血症"},
        "b": {"kws": ["hyponatremi", "sodium", "vasopressin", "water retention",
                      "desmopressin"],
              "ok": True,
              "note": "Desmopressin 为加压素类似物，水潴留→低钠血症"},
        "summary": "SSRI（Zimelidine）所致 SIADH 与 Desmopressin 的水潴留均致低钠，叠加增加低钠血症风险。"},
    "The risk or severity of hypotension and CNS depression increase": {
        "mech": ["hypotens", "vasodilat", "sedation", "CNS depressant"],
        "a": {"kws": ["hypotens", "sedat", "cns depress", "phenothiazine",
                      "alpha-adrenergic", "somnolence"],
              "ok": True,
              "note": "Thioridazine（吩噻嗪）α 受体阻滞→低血压，并具中枢抑制作用"},
        "b": {"kws": ["sedat", "cns depress", "somnolence", "drowsiness"],
              "ok": True,
              "note": "Naltrexone 具镇静/中枢作用（嗜睡为已知不良反应）"},
        "summary": "Thioridazine 的 α 受体阻滞（低血压）与两药的中枢抑制作用叠加，增加低血压与 CNS 抑制风险。"},
    "The risk or severity of electrolyte imbalance increase": {
        "mech": ["hypokalemia", "potassium", "electrolyte"],
        "a": {"kws": ["electrolyte", "potassium", "hypokalemi",
                      "mineralocorticoid"],
              "ok": True,
              "note": "糖皮质激素（Mometasone）大剂量有盐皮质激素样电解质作用"},
        "b": {"kws": ["hypokalemi", "potassium", "electrolyte", "thiazide"],
              "ok": True,
              "note": "噻嗪类利尿剂（Bendroflumethiazide）低钾/电解质紊乱为已知"},
        "summary": "噻嗪类利尿剂（Bendroflumethiazide）致低钾/电解质紊乱，糖皮质激素（Mometasone）的盐皮质激素样作用可加重。"},
    "The risk or severity of hyperthermia and oligohydrosis increase": {
        # keyword extension beyond the task vocabulary (recorded): the
        # oligohydrosis half of the event is best matched by sweating terms
        "mech": ["hyperthermia", "serotonin syndrome", "anhidrosis",
                 "hypohidrosis", "oligohydrosis", "sweating"],
        "a": {"kws": ["antimuscarinic", "sweat", "hyperthermia", "thermoregulat",
                      "anhidrosis"],
              "ok": True,
              "note": "Revefenacin 为长效抗胆碱（LAMA），抑制出汗→体温调节受损"},
        "b": {"kws": ["hyperthermia", "anhidrosis", "oligohydrosis", "sweat",
                      "thermoregulat", "hypohidrosis", "carbonic anhydrase"],
              "ok": True,
              "note": "Topiramate（碳酸酐酶抑制）报告少汗/高温"},
        "summary": "Topiramate（碳酸酐酶抑制）与 Revefenacin（抗胆碱）均损害出汗性体温调节，叠加增加高温与少汗风险。"},
}

_TIER_LABEL = {"direct": "direct（直接证据）",
               "class_suggested": "class_suggested（类别级机制建议，需人工裁决）",
               "none": "none（未识别）"}

# ---------------------------------------------------------------------------
# v1 evidence — HISTORICAL CONTEXT ONLY (controller ruling, 2026-08-23).
# The previous evidence pass (case_study_per_event.py, PubMed abstract
# co-occurrence, 2026-08-23) reported direct 2 / class_suggested 2, but on
# hard re-review both "direct" citations fail the strict standard (42312164
# is a vildagliptin protein-docking study — drug names in a drug list, no
# pair interaction; 21919844 is a herb-drug interaction drug list), and the
# two "class_suggested" records are not mechanism literature (42549822 NET
# occupancy model, 19393386 naltrexone side-effect observation). v2 tiers
# are therefore re-derived FROM ZERO by the two legs; v1 is shown per
# candidate in the markdown only as historical context. The fragments below
# are copied verbatim from external/outputs/case_evidence_dataset2_per_event.md.
# ---------------------------------------------------------------------------

V1_EVIDENCE = {
    "The risk or severity of angioedema increase": {
        "tier": "direct",
        "pmids": ["42312164"],
        "fragment": ("In silico analyses, including molecular docking as well as "
                     "molecular dynamics simulation, demonstrate good binding "
                     "affinity as well as stable interaction of vildagliptin with "
                     "PI3K (4YKN) and NLRP3 (7ALV) proteins in comparison to other "
                     "DPP-4 inhibitors (sitagliptin, saxagliptin, linagliptin, and "
                     "alogliptin)."),
        "note": "v1 判定（蛋白对接药物清单）不达 v2 direct 标准，仅作历史记录"},
    "The risk or severity of neutropenia increase": {
        "tier": "direct",
        "pmids": ["21919844"],
        "fragment": ("Drugs that have a high potential to interact with herbal "
                     "medicines usually have a narrow therapeutic index, including "
                     "warfarin, digoxin, cyclosporine, tacrolimus, amitriptyline, "
                     "midazolam, indinavir, and irinotecan."),
        "note": "v1 判定（草药相互作用高风险药清单）不达 v2 direct 标准，仅作历史记录"},
    "The serum concentration of the active metabolites increase": {
        "tier": "class_suggested",
        "pmids": ["42549822"],
        "fragment": ("To facilitate cross-drug comparisons of noradrenergic activity, "
                     "we developed a pharmacometric model to estimate NET occupancy "
                     "for 26 psychotropic agents and their active metabolites."),
        "note": "v1 判定（NET 占用建模，非机制文献）不达 v2 class_suggested 标准，仅作历史记录"},
    "The risk or severity of hypotension and CNS depression increase": {
        "tier": "class_suggested",
        "pmids": ["19393386"],
        "fragment": ("Certain side effects were observed, namely transitory sedation "
                     "at the beginning of treatment and moderate constipation."),
        "note": "v1 判定（不良反应观察，非机制文献）不达 v2 class_suggested 标准，仅作历史记录"},
}

_TIER_RANK = {"direct": 2, "class_suggested": 1, "none": 0}

# ---------------------------------------------------------------------------
# Manual adjudication (R29 authors' call, 2026-08-23): the automated legs
# (abstract drug-name + role-keyword co-occurrence, esummary title check)
# still surface citations whose abstracts mention the drug only incidentally
# or in a context that does NOT support the claimed mechanism/event
# direction. Each record below was manually reviewed against its REAL title
# and abstract fragments and is dropped with a recorded reason (宁缺毋滥 —
# prefer "none" over a forced label). Keys = event, values = {pmid: reason}.
# ---------------------------------------------------------------------------

ADJUDICATION_DROP = {
    "The absorption decrease": {
        "41471023": "3D 打印左旋多巴制剂论文：仅涉及新剂型生物利用度改进，"
                    "不含胃肠动力/吸收减少机制，与事件方向无关",
    },
    "the neuromuscular blocking activities decrease": {
        "40221998": "肌松药安全使用指南：仅确认 Pipecuronium 为非去极化肌松药，"
                    "摘要不含拟交感/β2 激动减弱神经肌肉阻滞的机制",
        "38949163": "酸碱平衡对肌松药影响综述：仅确认非去极化肌松药为烟碱受体竞争性"
                    "拮抗剂，不含 β2 激动剂减弱肌松的机制",
    },
    "The risk or severity of renal failure hypotension and hyperkalemia increase": {
        "42077598": "ACE 抑制剂 FAERS 分析：摘要仅有治疗用途陈述，无高钾/低血压/"
                    "肾损害证据",
        "39044930": "嘌呤霉素肾病动物模型：Quinapril 用于降低高血压（方向相反），"
                    "非 ACEi 不良反应机制文献",
    },
    "the stimulatory activities decrease": {
        "41732374": "右美沙芬/吡拉西坦 ADHD 单病例：刺激剂仅为背景提及，"
                    "非 Lisdexamfetamine 兴奋剂机制文献",
    },
    "The risk or severity of myopathy and weakness increase": {
        "42629679": "10 月龄儿童横纹肌溶解病例：Budesonide 为对症治疗（止咳雾化），"
                    "非肌病病因，不能支持类固醇肌病机制",
    },
    "The risk or severity of sedation and somnolence increase": {
        "41122869": "VMAT2 抑制剂综述：仅述 TD 治疗，摘要无镇静/嗜睡内容",
        "41782784": "Deutetrabenazine + MECT 病例报告：摘要无镇静/嗜睡内容",
    },
    "The risk or severity of neutropenia increase": {
        "42093871": "脂质体 vs 常规伊立替康安全性论文：摘要无中性粒细胞减少内容",
    },
    "an increase in the absorption resulting in an increased serum concentration and potentially a worsening of adverse effects cause": {
        "40816625": "OAT2 介导的葡萄糖醛酸外排研究：涉及肝内代谢处置，"
                    "非胃肠道吸收机制",
    },
    "The risk or severity of fluid retention increase": {
        "39309672": "HAE 患者 Danazol 真实世界研究：摘要中水肿为 HAE 疾病本身"
                    "发作表现，非 Danazol 所致液体潴留",
    },
    "The risk or severity of renal failure and hypertension increase": {
        "41909296": "免疫触须样肾小球病病例：高血压/肾功能受损为疾病基线，"
                    "非环孢素毒性",
        "41507552": "MALDI-MSI 肾脏沉积方法学研究：摘要未述肾毒性/高血压",
    },
    "the hypokalemic activities increase": {
        "41156295": "Indapamide 与氢氯噻嗪高钙尿症试验：摘要无低钾/钾相关内容",
    },
    "The risk or severity of hyponatremia increase": {
        "42597136": "经蝶术后迟发低钠血症研究：Desmopressin 仅为排除标准提及",
    },
}


# ---------------------------------------------------------------------------
# Leg B per-drug mechanism retrieval + direction-consistent support check
# ---------------------------------------------------------------------------

def mechanism_search(name, mech_kws):
    """'<drug>' AND (<kw> OR <kw> ...), relevance ordering. pageSize=10: the
    first 3 hits form the headline list reported in the markdown; the support
    check scans all 10 (documented extension — the top-3 alone frequently
    return incidental case reports)."""
    query = '"%s" AND (%s)' % (name, " OR ".join(mech_kws))
    return query, epmc_search(query, PAGE_B_SCAN)


def fragment_sentences(abstract, re_drug, kws):
    """Sentences containing any role keyword; drug-name sentences preferred.
    Returns up to 2 fragments, max 400 chars each."""
    norm = strip_tags(abstract)
    with_drug, without_drug = [], []
    for sent in _SENT_RE.split(norm):
        if any(k in sent.lower() for k in kws):
            (with_drug if re_drug.search(sent) else without_drug).append(sent)
    return (with_drug + without_drug)[:2]


def side_support(hits, name, role, mech_kws):
    """Does the retrieved mechanism literature support the claimed role?
    STRICT (controller ruling, 2026-08-23): an abstract counts when it
    contains the drug name AND any role keyword (direction-consistent,
    conservative) — AND the record's REAL title (re-fetched via esummary)
    must be consistent with the claim (contains the drug name/stem or a
    role/mechanism keyword), otherwise the citation is deleted. Preprints
    never count. Returns up to 2 verified records
    {'pmid','title','real_title','title_ok','fragments'}."""
    re_drug = _name_re(name)
    out = []
    for hit in hits:
        if NO_PREPRINT and hit.get("source") == "PPR":
            continue
        abst = strip_tags(hit.get("abstractText") or "")
        if not abst or not re_drug.search(abst):
            continue
        if not any(k in abst.lower() for k in role["kws"]):
            continue
        frags = fragment_sentences(abst, re_drug, role["kws"])
        if not frags:
            continue
        rec = {"pmid": hit_pmid(hit) or hit.get("id"),
               "title": hit.get("title", ""), "fragments": frags}
        rec = verify_record(rec, [name], role["kws"] + mech_kws)
        if not rec["title_ok"]:
            continue  # 硬校验：标题与证据主张不符 -> 删除该引用
        out.append(rec)
        if len(out) >= 2:
            break
    return out


# ---------------------------------------------------------------------------
# Per-candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_row(row):
    event = row["event"]
    a_raw = "" if pd.isna(row["a_name"]) else str(row["a_name"])
    b_raw = "" if pd.isna(row["b_name"]) else str(row["b_name"])
    a_name = a_raw or NAME_OVERRIDE.get(row["drug_a"], "")
    b_name = b_raw or NAME_OVERRIDE.get(row["drug_b"], "")
    a_term = query_term(a_name)
    b_term = query_term(b_name)
    # names never resolved at all -> "DrugBank DBxxxx" per controller ruling
    if not a_term:
        a_term = "DrugBank %s" % row["drug_a"]
    if not b_term:
        b_term = "DrugBank %s" % row["drug_b"]

    ev = {"event": event, "a_name": a_name or "(未解析)", "b_name": b_name or "(未解析)",
          "a_raw": a_raw or "", "b_raw": b_raw or "",
          "a_term": a_term, "b_term": b_term,
          "legA": {"q1": None, "hits1": [], "q2": None, "hits2": [],
                   "direct": None, "ft_checked": None, "cooccur_fulltext": None},
          "legB": {"queries": {}, "support": {"a": [], "b": []},
                   "hits": {"a": [], "b": []}},
          "tier": "none", "pmids": [], "note": ""}

    # ---- Leg A: full-text direct evidence -------------------------------
    q1 = '("%s" AND "%s") AND (interaction OR adverse OR effect)' % (a_term, b_term)
    q2 = '("%s" AND "%s")' % (a_term, b_term)
    ev["legA"]["q1"], ev["legA"]["q2"] = q1, q2
    try:
        ev["legA"]["hits1"] = epmc_search(q1, PAGE_A)
        ev["legA"]["hits2"] = epmc_search(q2, PAGE_A_WIDE)
    except RuntimeError:
        pass  # network failure for this query: recorded as empty

    # Same-drug pair (a_term == b_term): a single occurrence of the name
    # trivially satisfies both sides of the co-occurrence rule, so the
    # two-drug direct tier is skipped entirely (documented decision) — the
    # mechanism leg (class-level) governs such candidates.
    same_drug = (a_term == b_term)
    direct = None
    for hit in ev["legA"]["hits1"] + ev["legA"]["hits2"]:
        if NO_PREPRINT and hit.get("source") == "PPR":
            continue
        if same_drug:
            break
        abst = hit.get("abstractText") or ""
        direct = direct_check(abst, a_term, b_term)
        if direct:
            # direct 硬校验：仅当同一句/标题同时讨论两药且含相互作用语境；
            # 再以 esummary 拉取真实标题，标题必须含其中一药名或相互作用词，
            # 否则（如药物清单、蛋白对接语境）不达 v2 direct 标准，删除。
            rec = {"pmid": hit_pmid(hit) or hit.get("id"),
                   "title": strip_tags(hit.get("title", "")),
                   "fragment": direct[0], "kw": direct[1], "where": "abstract"}
            rec = verify_record(rec, [a_term, b_term], DIRECT_KW)
            if rec["title_ok"]:
                ev["legA"]["direct"] = rec
                direct = rec
            else:
                direct = None
            break
    if direct is None and not same_drug:
        # full-text check: first hit with PMC full text (at most one fetch);
        # only the abstract+body text is inspected, and only same-sentence
        # co-occurrence counts on full text (conservative). Full-text
        # co-occurrence alone never drives the direct tier (verifiability
        # from title/abstract) — it is reported as a co-occurrence probe.
        for hit in ev["legA"]["hits1"] + ev["legA"]["hits2"]:
            if NO_PREPRINT and hit.get("source") == "PPR":
                continue
            pmcid = hit.get("pmcid") or (hit.get("id")
                                         if hit.get("source") == "PMC" else "")
            if pmcid and hit.get("inPMC") == "Y":
                try:
                    xml = epmc_fulltext_xml(pmcid)
                except RuntimeError:
                    break
                body = fulltext_body_text(xml)
                ev["legA"]["ft_checked"] = pmcid
                ev["legA"]["cooccur_fulltext"] = bool(
                    _name_re(a_term).search(body) and _name_re(b_term).search(body))
                d2 = direct_check(body, a_term, b_term,
                                  same_sentence_only=True)
                if d2:
                    rec = {"pmid": hit_pmid(hit) or hit.get("id"),
                           "title": strip_tags(hit.get("title", "")),
                           "fragment": d2[0], "kw": d2[1], "where": "fulltext"}
                    rec = verify_record(rec, [a_term, b_term], DIRECT_KW)
                    if rec["title_ok"]:
                        ev["legA"]["direct"] = rec
                        direct = rec
                break
    if direct:
        ev["tier"] = "direct"
        ev["pmids"].append(ev["legA"]["direct"]["pmid"])

    # ---- Leg B: class-level mechanism evidence --------------------------
    role = ROLES.get(event)
    if role is not None:
        mech_kws = role["mech"]
        for side, (raw, term, rside) in (("a", (a_raw, a_term, role["a"])),
                                         ("b", (b_raw, b_term, role["b"]))):
            if not term:
                continue
            q, hits = mechanism_search(term, mech_kws)
            ev["legB"]["queries"][side] = q
            ev["legB"]["hits"][side] = [
                {"pmid": hit_pmid(hit) or hit.get("id"),
                 "title": strip_tags(hit.get("title", "")),
                 "has_abstract": bool(hit.get("abstractText"))}
                for hit in hits]
            # same drug on both sides -> reuse the identical query results
            other = "b" if side == "a" else "a"
            if ev["legB"]["queries"].get(other) == q and \
                    ev["legB"]["support"].get(other):
                ev["legB"]["support"][side] = ev["legB"]["support"][other]
            else:
                ev["legB"]["support"][side] = side_support(hits, term, rside,
                                                           mech_kws)
            # manual adjudication: drop citations whose fragment context does
            # not truly support the claim (recorded reasons, 宁缺毋滥)
            adj = ADJUDICATION_DROP.get(event, {})
            dropped = [r["pmid"] for r in ev["legB"]["support"][side]
                       if r["pmid"] in adj]
            if dropped:
                ev.setdefault("adjudication", {})[side] = {
                    p: adj[p] for p in dropped}
                ev["legB"]["support"][side] = [
                    r for r in ev["legB"]["support"][side]
                    if r["pmid"] not in adj]
        if ev["tier"] == "none":
            supported = [s for s, recs in ev["legB"]["support"].items() if recs]
            consistent = [s for s in supported if role[s]["ok"]]
            if consistent:
                ev["tier"] = "class_suggested"
                for s in sorted(consistent):
                    ev["pmids"].extend(r["pmid"] for r in ev["legB"]["support"][s])
    # evidence_note: every sentence must be traceable to the cited records
    # (controller ruling). For class_suggested the note names the supported
    # side(s), the claim as it appears in the verified titles/abstracts, and
    # the supporting PMIDs; the class-level inference is labelled as such.
    # For direct it names the pair-interaction record; the mechanism sentence
    # is appended only when the fragment itself shows the mechanism keyword.
    if ev["tier"] == "direct":
        d = ev["legA"]["direct"]
        frag = d["fragment"][:160]
        mech = ""
        if role is not None and any(
                k in frag.lower() for k in role["mech"]):
            mech = "；机制：%s" % role["summary"]
        ev["note"] = ("两药在同一文献（PMID %s，标题“%s”）的摘要中共同出现且含相互"
                      "作用语境词（%s）。证据片段：“%s”%s"
                      % (d["pmid"], (d.get("real_title") or d["title"])[:150],
                         d["kw"], frag, mech))
    elif ev["tier"] == "class_suggested":
        parts = []
        for s in ("a", "b"):
            recs = ev["legB"]["support"].get(s, [])
            if not recs or not role[s]["ok"]:
                continue
            pms = "/".join(r["pmid"] for r in recs)
            t1 = (recs[0].get("real_title") or recs[0]["title"])[:120]
            parts.append("%s（文献%s，标题“%s”）" % (
                role[s]["note"], pms, t1))
        if parts:
            # the composite summary is appended only when BOTH sides have
            # surviving citations — otherwise it would claim mechanisms the
            # evidence does not cover (traceability rule)
            both = all(recs for recs in ev["legB"]["support"].values())
            ev["note"] = "；".join(parts) + "。" + \
                (role["summary"] if both else "") + \
                "（类别级推断：所引为单药机制文献，非两药合用直接证据。）"
        else:
            ev["note"] = "未检索到支持该事件方向的 direct/类别级机制证据"
    else:
        ev["note"] = "未检索到支持该事件方向的 direct/类别级机制证据"
    # dedupe, keep order
    seen, pmids = set(), []
    for p in ev["pmids"]:
        if p and p not in seen:
            seen.add(p)
            pmids.append(p)
    ev["pmids"] = pmids
    return ev


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_md(cands, evs):
    n = len(evs)
    n_direct = sum(1 for e in evs if e["tier"] == "direct")
    n_class = sum(1 for e in evs if e["tier"] == "class_suggested")
    n_none = n - n_direct - n_class
    n_legA_hits = sum(1 for e in evs if e["legA"]["direct"])
    n_legB_sup = sum(1 for e in evs
                     if e["legB"]["support"]["a"] or e["legB"]["support"]["b"])
    lines = [
        "# Case-Study Evidence Upgrade (Dataset 2, Task 17): 全文检索 + 类别级机制证据",
        "",
        f"- 生成日期: {datetime.date.today().isoformat()}",
        f"- 输入: `{os.path.basename(CAND_CSV)}`（25 候选，v1 三档: "
        "direct 2 / class_suggested 2 / none 21）",
        f"- **v2 三档统计: direct {n_direct} / class_suggested {n_class} / "
        f"none {n_none}（共 {n} 候选）**",
        f"- 腿 A 直接证据命中（标题/摘要同时讨论两药 + 相互作用语境）: {n_legA_hits} 候选",
        f"- 腿 B 类别级机制支持（≥1 侧方向一致且标题硬校验通过）: {n_legB_sup} 候选",
        "- 检索源: Europe PMC REST（`search` + `fullTextXML`），resultType=core，"
        "relevance 相关度排序（EPMC 默认；能浮出经典机制文献，缓存冻结结果保证可复现）；"
        "0.5 s/请求限速（≥0.4 s），失败自动重试 3 次；"
        "全部响应缓存于 `case_evidence_upgrade_cache.json`（可离线复现）。",
        "- **v1 档位（历史记录，不再沿用）**: v1（PubMed 摘要共现）报告的 "
        "direct 2 / class_suggested 2 经硬复核均不达 v2 标准——42312164 为蛋白"
        "对接研究（药名仅出现在药物清单）、21919844 为草药相互作用风险药清单、"
        "42549822 为 NET 占用建模、19393386 为不良反应观察——v2 三档由两腿从零"
        "重新判定，v1 仅在各候选节内作历史标注（PMID 与片段逐字引自 "
        "`case_evidence_dataset2_per_event.md`）。",
        "- **硬校验（每条证据引用）**: evidence_pmids 中的每个 PMID 均通过 NCBI "
        "eutils esummary 重新拉取真实标题；标题与证据主张不符的引用一律删除"
        "（宁缺毋滥）。摘要命中后仍须标题可证，无 PMID 的 PMC 记录以 EPMC 核心"
        "元数据标题为准。",
        "- **人工裁决（R29）**: 自动腿命中后逐条对照真实标题与摘要片段复核；片段"
        "仅属偶然提及（如作为对症治疗、排除标准、疾病基线、药物清单）或摘要不含"
        "所主张机制的引用一律删除并记录原因（见各候选节“人工裁决剔除”）。",
        "- **三档定义（严格执行）**: **direct** = 同一文献的标题/摘要中**同时"
        "讨论两药**且含相互作用语境（同一句共现 + 交互/不良反应词；全文共现只作"
        "探测不作判档）；**class_suggested** = **单药机制文献**，摘要含药名 + "
        "方向一致机制词，且真实标题与主张一致（每条附标题原文与片段）；"
        "**none** = 其余。",
        "- **腿 A（全文检索升级）**: q1 = `(\"name_a\" AND \"name_b\") AND "
        "(interaction OR adverse OR effect)`（pageSize=5，记 PMID/标题/"
        "inEPMC/inPMC 全文标志）；q2 = `(\"name_a\" AND \"name_b\")`（pageSize=3，"
        "宽共现探测）。",
        "- **腿 B（类别级机制，按事件方向）**: 对两药分别检索 "
        "`\"<drug>\" AND (<机制关键词>)`（pageSize=10：头条 top-3 列于节内，"
        "支持判定扫描全部 10 篇——top-3 常为无关病例报告，此为记录的扩展）；"
        "机制关键词按事件文本从任务词表选定（QTc/吸收/代谢/排泄/低钾/低钠/"
        "中性粒/肌病/镇静/神经肌肉/血栓/液体潴留/血管性水肿/刺激/高温等，"
        "高温事件扩展了 anhidrosis/hypohidrosis/oligohydrosis/sweating 并记录）。"
        "**判定规则**: 药物的机制文献与该药物在事件方向中的作用一致（如事件"
        "\"血压降低\"中任一药为血管舒张/降压机制，或另一药为升压剂被拮抗）→ "
        "class_suggested，附双方各 1-2 篇 PMID + 标题原文 + 证据片段 + 一句机制"
        "解释。判定保守：宁可不标，不得瞎标。Preprint（PPR）不参与证据档判定。",
        "- 证据档优先级: direct > class_suggested > none。自动结果仅为人工复核"
        "材料；最终 Evidence 列由作者裁决（R29）。",
        "- 药名: 未解析的 DB id 使用 go.drugbank.com 解析名"
        f"（{ {v: k for k, v in NAME_OVERRIDE.items()} }），解析失败才回退 "
        "`DrugBank {DB_ID}`；原始名/检索词均如实记录。",
        "",
    ]
    for c in cands.itertuples(index=False):
        e = next(x for x in evs if x["event"] == c.event)
        v1 = V1_EVIDENCE.get(c.event)
        lines += [
            f"## {e['a_name']} + {e['b_name']}（{c.event}）",
            f"- 药对: `{c.drug_a}` / `{c.drug_b}`; a_name={e['a_raw'] or '(未解析)'}, "
            f"b_name={e['b_raw'] or '(未解析)'}; 检索词: `{e['a_term']}` / `{e['b_term']}`",
            f"- 事件: {c.event}; semantic_overlap: {c.semantic_overlap}; "
            f"prob_mean={c.prob_mean:.4f}, r={c.r:.4f}",
            f"- v1 证据档: {_TIER_LABEL[v1['tier']] if v1 else 'none（未识别）'}"
            + (" → **v2 证据档: %s**" % _TIER_LABEL[e["tier"]]),
            f"- 关键 PMID: {', '.join(e['pmids']) or '（无）'}",
            f"- 机制解释（一句话）: {e['note']}",
            "- 检索式记录:",
            f"  - 腿 A q1: `{e['legA']['q1']}`",
            f"  - 腿 A q2: `{e['legA']['q2']}`",
            "  - 腿 B: " + " / ".join(
                f"`{q}`" for q in e["legB"]["queries"].values()) or "  - （腿 B 未执行）",
            "- **腿 A（Europe PMC 全文检索）** q1 top-5:",
        ]
        if not e["legA"]["hits1"]:
            lines.append("  - （无结果）")
        for hit in e["legA"]["hits1"]:
            pm = hit_pmid(hit) or hit.get("id")
            ft = ("全文: EPMC/inPMC" if hit.get("inEPMC") == "Y"
                  else ("全文: PMC" if hit.get("inPMC") == "Y" else "无全文"))
            lines.append(f"  - [{pm}](https://pubmed.ncbi.nlm.nih.gov/{pm}/) — "
                         f"{strip_tags(hit.get('title', ''))}（{ft}）")
        lines.append("  - q2 top-3（宽共现探测）:")
        if not e["legA"]["hits2"]:
            lines.append("    - （无结果）")
        for hit in e["legA"]["hits2"]:
            pm = hit_pmid(hit) or hit.get("id")
            lines.append(f"    - [{pm}](https://pubmed.ncbi.nlm.nih.gov/{pm}/) — "
                         f"{strip_tags(hit.get('title', ''))}")
        if e["legA"]["ft_checked"]:
            co = "两药在正文中共现" if e["legA"]["cooccur_fulltext"] else "两药在正文中未共现"
            lines.append(f"  - 全文检查: PMC 全文（{e['legA']['ft_checked']}）已抓取，"
                         f"{co}；Direct 判定见下")
        if e["legA"]["direct"]:
            d = e["legA"]["direct"]
            lines.append(f"  - **Direct 证据（{d['where']}，PMID {d['pmid']}，"
                         f"语境词 {d['kw']}）**: 真实标题（esummary 硬校验）: "
                         f"“{(d.get('real_title') or d['title'])[:200]}”")
            lines.append(f"    - 证据片段: “{d['fragment'][:400]}”")
        else:
            lines.append("  - 未检出 direct 证据（两药在标题/摘要中无同时"
                         "讨论且含相互作用语境的记录）")
        lines.append("- **腿 B（类别级机制，按事件方向）**:")
        for side, label in (("a", f"{e['a_name']} ({e['a_term']})"),
                            ("b", f"{e['b_name']} ({e['b_term']})")):
            raw_hits = e["legB"]["hits"][side]
            lines.append(f"  - {label}: 检索 top-{len(raw_hits) or 0}"
                         + ("" if raw_hits else "（无结果）"))
            for h in raw_hits:
                lines.append(f"    - [{h['pmid']}](https://pubmed.ncbi.nlm.nih.gov/"
                             f"{h['pmid']}/) — {h['title']}"
                             + ("（有摘要）" if h["has_abstract"] else "（无摘要）"))
            sup = e["legB"]["support"][side]
            if sup:
                lines.append(f"    - 方向一致且标题硬校验通过 {len(sup)} 篇:")
                for r in sup:
                    lines.append(f"      - PMID {r['pmid']} — 真实标题（esummary）: "
                                 f"“{(r.get('real_title') or r['title'])[:200]}”")
                    for i, frag in enumerate(r["fragments"], 1):
                        lines.append(f"        - 片段 {i}: “{frag[:300]}”")
            else:
                lines.append("    - （无方向一致的机制文献支持；命中但标题硬校验"
                             "不通过/人工裁决剔除的引用已删除）")
        adj = e.get("adjudication", {}).get(side)
        if adj:
            lines.append(f"    - **人工裁决剔除（{side} 侧）**:")
            for p, reason in adj.items():
                lines.append(f"      - PMID {p}: {reason}")
        v1 = V1_EVIDENCE.get(c.event)
        if v1 is not None:
            lines.append(f"- **v1 历史档位（不沿用）**: {_TIER_LABEL[v1['tier']]}，"
                         f"PMID {', '.join(v1['pmids'])}: “{v1['fragment'][:200]}”")
        lines.append("")
    return "\n".join(lines)


def write_csv(cands, evs, path):
    rows = []
    for c in cands.itertuples(index=False):
        e = next(x for x in evs if x["event"] == c.event)
        rows.append({
            "rank": c.rank, "event": c.event, "drug_a": c.drug_a,
            "drug_b": c.drug_b, "a_name": c.a_name, "b_name": c.b_name,
            "prob_mean": c.prob_mean, "u_mean": c.u_mean, "r": c.r,
            "semantic_overlap": c.semantic_overlap,
            "evidence_auto": e["tier"],
            "evidence_pmids": ", ".join(e["pmids"]),
            "evidence_note": e["note"],
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig",
                              quoting=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N candidates (dev)")
    args = ap.parse_args(argv)

    cands = pd.read_csv(CAND_CSV)
    if args.limit:
        cands = cands.head(args.limit)
    evs = [evaluate_row(row) for row in cands.to_dict("records")]
    md = render_md(cands, evs)
    with open(V2_MD, "w", encoding="utf-8") as f:
        f.write(md)
    write_csv(cands, evs, V2_CSV)
    _save_cache()
    n = len(evs)
    print(f"wrote {V2_MD} ({len(md)} chars) and {V2_CSV}")
    print("tiers:", {t: sum(1 for e in evs if e['tier'] == t)
                     for t in ("direct", "class_suggested", "none")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
