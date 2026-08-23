#!/usr/bin/env python
# coding=utf-8
"""Task 16: Dataset-2 case study — per-event best candidate + three-tier evidence
(R28 / R29). Final deliverable of the external-validation case study.

Controller rulings implemented here (binding):

- R28: for EACH of the 25 test2 events take exactly ONE best candidate = the
  (drug_a, drug_b, event) triple with y_true=1 and the highest r within that
  event (r = p_mean * (1 - u_mean), 5-seed means), excluding pairs overlapping
  Dataset 1 (dataset2_pair_overlap.json, reused from audit_overlap_dataset2).
  25 events -> at most 25 rows; ALL rows are reported — no picking, no
  dropping. This replaces the earlier global top-20 rule.
- R24/R12: report everything truthfully; FAERS does not count as independent
  evidence (and there is no FAERS data for dataset2 — no FAERS columns).
- R26: every row carries the semantic_overlap flag: "no" iff the event is one
  of the 10 semantic-disjoint events (max cosine to Dataset 1 < 0.7,
  external/outputs/disjoint_events.json), else "yes".
- R29 (three-tier evidence, automated part). Per candidate:
  1. PubMed esearch `"{name_a}"[All Fields] AND "{name_b}"[All Fields]`
     (retmax 5, 0.4 s/request rate limit), esummary for titles, efetch for
     abstract texts.
  2. **Direct**: in ANY abstract both drug names (query terms) co-occur in the
     same sentence OR within a +/-150-char window, and the window contains an
     interaction/adverse/effect-class word (conservative vocabulary, mirrors
     the reviewed case_study_ext CONTEXT_KW plus effect-word variants).
  3. **Class-level (suggested tier)**: abstract contains at least one drug
     name AND hits mechanism words related to the event direction
     (vasorelax*/vasodilat*/hypotens*/blood pressure; absorption; metabolism;
     serum/plasma concentration; sedat*/CNS depress* etc. — per-event word
     table below, chosen from the event text). Matching sentences are output
     as fragments for human adjudication.
  4. Everything else = Not identified ("none").
  The automated result is REVIEW MATERIAL ONLY; the final Evidence column is
  the authors' call.

Reuse decisions (vs copying; follows the reviewed case_study_dataset2.py T12
and case_study_dataset2_final.py T15):
  - cs.load_predictions / cs.aggregate_candidates / cs.load_excluded_pairs /
    cs.resolve_names imported unchanged (resolve_names memoized via lru_cache,
    identical to the T15 final script).
  - case_study_ext._eutils_get / parse_esearch_ids / parse_esummary_titles /
    CONTEXT_KW imported unchanged.
  - Everything else (per-event selection, efetch XML parsing, three-tier
    evaluation, rendering) is new here.

Outputs:
  external/outputs/case_candidates_dataset2_per_event.csv
  external/outputs/case_evidence_dataset2_per_event.md
"""
import argparse
import datetime
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from functools import lru_cache

import pandas as pd

import case_study_dataset2 as cs
from case_study_ext import (CONTEXT_KW, _eutils_get, parse_esearch_ids,
                            parse_esummary_titles)

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(OUT, ".."))
OUTDIR = os.path.join(OUT, "outputs")
PRED_CSV = os.path.join(OUTDIR, "predictions_ds2_retrained_0shot.csv")
DISJOINT_JSON = os.path.join(OUTDIR, "disjoint_events.json")
TEST2_TASKS = os.path.join(REPO, "EviDDIE", "dataset2", "test2_tasks.json")
CAND_CSV = os.path.join(OUTDIR, "case_candidates_dataset2_per_event.csv")
EVID_MD = os.path.join(OUTDIR, "case_evidence_dataset2_per_event.md")
TIER = "test2"
RETMAX = 5
RATE_LIMIT_S = 0.4
WINDOW_CHARS = 150
N_EVENTS = 25

CAND_COLUMNS = ["rank", "event", "drug_a", "drug_b", "a_name", "b_name",
                "prob_mean", "u_mean", "r", "semantic_overlap", "evidence_auto"]

# Direct-tier context words (R29, conservative): mirrors the reviewed
# CONTEXT_KW (T10/T12) plus a few effect-word variants. A window must
# literally contain one of these (substring, case-insensitive).
DIRECT_KW = CONTEXT_KW + ("additive", "synergis", "potentiat")

# Class-level mechanism keywords per event (R29): chosen from the event text
# plus the standard mechanism vocabulary; plain substrings, matched
# case-insensitively against the abstract text.
MECH_KW = {
    "The absorption decrease": ["absorb", "bioavailab"],
    "The risk or severity of QTc prolongation and hypotension increase":
        ["qtc", "qt interval", "torsad", "hypotens", "blood pressure"],
    "The risk or severity of QTc prolongation decrease":
        ["qtc", "qt interval", "torsad", "prolong"],
    "The risk or severity of Tachycardia and drowsiness increase":
        ["tachycard", "drowsiness", "somnolence", "sedat"],
    "The risk or severity of angioedema increase": ["angioedema"],
    "The risk or severity of electrolyte imbalance increase": ["electrolyte"],
    "The risk or severity of fluid retention increase":
        ["fluid retention", "edema", "oedema"],
    "The risk or severity of hypertension decrease":
        ["hypotens", "blood pressure", "vasodilat", "vasorelax"],
    "The risk or severity of hyperthermia and oligohydrosis increase":
        ["hyperthermia", "thermoregulat", "body temperature", "sweat"],
    "The risk or severity of hyponatremia increase":
        ["hyponatremi", "serum sodium", "vasopressin"],
    "The risk or severity of hypotension and CNS depression increase":
        ["hypotens", "cns depress", "sedat", "blood pressure"],
    "The risk or severity of myopathy and weakness increase":
        ["myopath", "rhabdomyolysis", "weakness", "muscl"],
    "The risk or severity of neutropenia and thrombocytopenia increase":
        ["neutropeni", "thrombocytopeni"],
    "The risk or severity of neutropenia increase": ["neutropeni"],
    "The risk or severity of renal failure and hypertension increase":
        ["renal fail", "kidney", "nephrotoxic", "creatinin", "blood pressure",
         "hypotens"],
    "The risk or severity of renal failure hypotension and hyperkalemia increase":
        ["renal fail", "kidney", "hyperkalemi", "potassium", "hypotens"],
    "The risk or severity of renal failure increase":
        ["renal fail", "kidney", "nephrotoxic", "creatinin"],
    "The risk or severity of sedation and somnolence increase":
        ["sedat", "somnolence", "cns depress", "hypnotic"],
    "The serum concentration of the active metabolites increase":
        ["active metabolite", "serum concentration", "plasma concentration",
         "metaboli"],
    "an increase in the absorption resulting in an increased serum concentration and potentially a worsening of adverse effects cause":
        ["absorb", "serum concentration", "plasma concentration", "bioavailab"],
    "the hypokalemic activities increase":
        ["hypokalemi", "serum potassium", "potassium"],
    "the neuromuscular blocking activities decrease":
        ["neuromuscular block", "muscle relax", "curar"],
    "the stimulatory activities decrease": ["stimulat"],
    "the thrombogenic activities increase":
        ["thrombogen", "platelet aggregat", "thrombosis"],
    "the vasopressor activities increase":
        ["vasopressor", "vasoconstrict", "pressor", "blood pressure"],
}

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Selection (R28): per-event best candidate
# ---------------------------------------------------------------------------

def select_per_event(pred, overlap_pairs, test2_events):
    """Per test2 event, the single best (drug_a, drug_b, event) candidate:
    highest r among y_true=1 triples not overlapping Dataset 1 at pair level.
    Tie-break on (drug_a, drug_b) for platform-stable output (determinism
    rule). rank = within-event rank (always 1). Returns the DataFrame sorted
    by r descending."""
    agg = cs.aggregate_candidates(pred)
    mask = pd.Series([tuple(sorted((a, b))) in overlap_pairs
                      for a, b in zip(agg["drug_a"], agg["drug_b"])],
                     index=agg.index)
    pool = agg.loc[~mask].copy()
    pool = pool.sort_values(["r", "drug_a", "drug_b"],
                            ascending=[False, True, True])
    best = pool.groupby("event", sort=False).first().reset_index()
    missing = set(test2_events) - set(best["event"])
    if missing:
        raise RuntimeError(
            f"events without any eligible candidate after pair exclusion: "
            f"{sorted(missing)} (R28: all 25 events must be listed)")
    best["rank"] = 1
    best = best.sort_values(["r", "event"], ascending=[False, True]).reset_index(drop=True)
    # within-event optimality invariant: each candidate must carry its
    # event's maximum r over the eligible pool.
    max_r = pool.groupby("event")["r"].max()
    dev = best["r"] - best["event"].map(max_r).astype(float)
    if (dev.abs() > 1e-9).any():
        raise RuntimeError("per-event optimum violated (selection bug)")
    return best[["rank", "event", "drug_a", "drug_b", "prob_mean", "u_mean", "r"]]


# ---------------------------------------------------------------------------
# PubMed retrieval (R29): esearch -> esummary titles -> efetch abstracts
# ---------------------------------------------------------------------------

def _eutils_get_retry(params, endpoint="esearch.fcgi", attempts=3, backoff=2.0):
    last = None
    for i in range(attempts):
        try:
            return _eutils_get(params, endpoint=endpoint)
        except Exception as exc:  # network / HTTP errors
            last = exc
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
    raise RuntimeError(f"eutils {endpoint} failed after {attempts} attempts: {last}")


def parse_efetch_abstracts(xml_text):
    """efetch XML -> {pmid: abstract text} (PubMed abstract paragraphs joined)."""
    root = ET.fromstring(xml_text)
    out = {}
    for art in root.iter():
        if not art.tag.endswith("PubmedArticle"):
            continue
        pmid, texts = None, []
        for node in art.iter():
            if node.tag.endswith("PMID") and pmid is None:
                pmid = (node.text or "").strip()
            elif node.tag.endswith("AbstractText"):
                texts.append("".join(node.itertext()).strip())
        if pmid and texts:
            joined = " ".join(t for t in texts if t)
            if joined:
                out[pmid] = joined
    return out


def pubmed_retrieve(a_term, b_term, retmax=RETMAX, sleep=RATE_LIMIT_S):
    """esearch (both names only, R29) -> esummary titles -> efetch abstracts.
    Returns {'pmids': [...], 'titles': {pmid: title}, 'abstracts': {pmid: text}}."""
    term = f'"{a_term}"[All Fields] AND "{b_term}"[All Fields]'
    body = _eutils_get_retry({"db": "pubmed", "retmode": "json", "retmax": retmax,
                              "term": term, "tool": "PharDDIE-ext-validation"})
    time.sleep(sleep)
    pmids = parse_esearch_ids(body)
    titles, abstracts = {}, {}
    if pmids:
        body2 = _eutils_get_retry({"db": "pubmed", "retmode": "json",
                                   "id": ",".join(pmids),
                                   "tool": "PharDDIE-ext-validation"},
                                  endpoint="esummary.fcgi")
        time.sleep(sleep)
        titles = parse_esummary_titles(body2)
        body3 = _eutils_get_retry({"db": "pubmed", "retmode": "xml",
                                   "id": ",".join(pmids),
                                   "tool": "PharDDIE-ext-validation"},
                                  endpoint="efetch.fcgi")
        time.sleep(sleep)
        abstracts = parse_efetch_abstracts(body3)
    return {"pmids": pmids, "titles": titles, "abstracts": abstracts}


# ---------------------------------------------------------------------------
# Three-tier evaluation (R29): direct > class_suggested > none
# ---------------------------------------------------------------------------

def _name_re(name):
    """Case-insensitive word-boundary matcher for a drug name, tolerant of a
    simple plural suffix. Word boundaries are ASCII-alphanumeric lookarounds so
    hyphenated/parsed names still match."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) +
                      r"(?:s|es)?(?![A-Za-z0-9])", re.IGNORECASE)


def _normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def direct_check(abstract, a_term, b_term):
    """R29 Direct tier: both query terms co-occur in one sentence or a
    +/-150-char window that also contains an interaction/adverse/effect-class
    word. Returns the matching fragment, or None."""
    if not a_term or not b_term:
        return None
    ra, rb = _name_re(a_term), _name_re(b_term)
    pos_a = [m.start() for m in ra.finditer(abstract)]
    pos_b = [m.start() for m in rb.finditer(abstract)]
    if not pos_a or not pos_b:
        return None
    text = _normalize(abstract)
    for sent in _SENT_RE.split(text):
        if ra.search(sent) and rb.search(sent) and \
                any(k in sent.lower() for k in DIRECT_KW):
            return sent
    for pa in pos_a:
        for pb in pos_b:
            if abs(pa - pb) <= WINDOW_CHARS:
                lo = max(0, min(pa, pb) - WINDOW_CHARS)
                hi = min(len(abstract), max(pa, pb) + WINDOW_CHARS)
                span = _normalize(abstract[lo:hi])
                if any(k in span.lower() for k in DIRECT_KW):
                    return span
    return None


def class_check(abstract, a_term, b_term, mech_kws):
    """R29 class-level tier: abstract contains at least one drug name AND hits
    mechanism keywords; returns (matched_keywords, fragments) where fragments
    are the matching sentences (drug-name sentences preferred, max 2)."""
    if not a_term or not b_term:
        return [], []
    ra, rb = _name_re(a_term), _name_re(b_term)
    if not (ra.search(abstract) or rb.search(abstract)):
        return [], []
    low = abstract.lower()
    matched = [k for k in mech_kws if k in low]
    if not matched:
        return [], []
    with_drug, without_drug = [], []
    for sent in _SENT_RE.split(_normalize(abstract)):
        if any(k in sent.lower() for k in matched):
            (with_drug if (ra.search(sent) or rb.search(sent))
             else without_drug).append(sent)
    return matched, (with_drug + without_drug)[:2]


def evaluate_evidence(abstracts, a_term, b_term, mech_kws):
    """Three tiers per R29: direct wins, else class_suggested, else none.
    Returns (tier, detail)."""
    for pmid in sorted(abstracts):
        frag = direct_check(abstracts[pmid], a_term, b_term)
        if frag:
            return "direct", {"pmid": pmid, "fragment": frag}
    for pmid in sorted(abstracts):
        kws, frags = class_check(abstracts[pmid], a_term, b_term, mech_kws)
        if frags:
            return "class_suggested", {"pmid": pmid, "keywords": kws,
                                       "fragments": frags}
    return "none", {}


# ---------------------------------------------------------------------------
# Rendering (final md: R28/R29/R26 framing, per-event sections)
# ---------------------------------------------------------------------------

_TIER_LABEL = {"direct": "direct（直接证据）",
               "class_suggested": "class_suggested（类别级机制建议，需人工裁决）",
               "none": "none（未识别）"}


def _tier_rationale(ev):
    tier, d = ev["tier"], ev["detail"]
    if tier == "direct":
        return (f"摘要 {d['pmid']} 中两药查询词出现在同一句/±150 字符窗口内且含"
                f"交互/不良反应语境词（{DIRECT_KW}）；片段: “{d['fragment'][:400]}”")
    if tier == "class_suggested":
        return (f"摘要 {d['pmid']} 含至少一个药名且命中事件机制词 {d['keywords']}；"
                f"命中句片段见下（供人工裁决）")
    n = len(ev["abstracts"])
    if not ev["pmids"]:
        return "PubMed 检索无结果（两药名联合查询无命中）"
    return (f"检索到 {len(ev['pmids'])} 篇，其中 {n} 篇有摘要；均未通过 Direct "
            "判定，也未见药名+机制词共现 → Not identified")


def render_md(cands, evidence, src_csv, n_overlap, disjoint_events, test2_events):
    n = len(evidence)
    n_direct = sum(1 for e in evidence if e["tier"] == "direct")
    n_class = sum(1 for e in evidence if e["tier"] == "class_suggested")
    n_none = n - n_direct - n_class
    n_no_ov = sum(1 for e in evidence if e["semantic_overlap"] == "no")
    lines = [
        "# Case-Study Evidence (Dataset 2, 每事件最优候选 + 三档证据): 人工复核材料",
        "",
        f"- 生成日期: {datetime.date.today().isoformat()}",
        f"- 来源预测 CSV: `{os.path.basename(src_csv)}`（tier=test2, y_true=1, "
        "5 训练种子: 19940419 / 20230801 / 20240520 / 20260201 / 20260301，Dataset 2 上重训）",
        f"- 选择规则（R28）: 每个 test2 事件取 **1 个最优候选** = 该事件内 y_true=1、"
        "r 最高（r = p_mean·(1−u_mean)，5 种子均值）的 (drug_a, drug_b, event) 三元组；"
        f"剔除与 Dataset 1 药对级重叠的 {n_overlap} 个 test2 药对（dataset2_pair_overlap.json）。"
        f"**{n}/{len(test2_events)} 个事件均有候选 → {n} 行，全部列出，无挑选、无剔除。**",
        f"- 药名解析: DB ID → drug_smiles → InChIKey-14 → RxPairEvid 名字表"
        f"（ddi_pairs_50k.csv）；未解析的药对用 `DrugBank {{DB_ID}}` 作 PubMed 查询词"
        "（dataset2 无名字文件，controller ruling b）。",
        f"- **semantic_overlap 标志（R26）**: “yes” = 该候选事件与 Dataset 1 某事件 "
        f"max cosine ≥ 0.7（不在 disjoint_events.json）；“no” = 事件在 10 个语义不相交"
        f"事件中（cos < 0.7）。本表 {n_no_ov}/{n} 个候选为 “no”。",
        f"- **三档证据统计（R29，自动判定，人工复核材料）: "
        f"direct {n_direct} / class_suggested {n_class} / none {n_none}（共 {n} 候选）**",
        "- 检索: `\"a_name\"[All Fields] AND \"b_name\"[All Fields]`（retmax 5；"
        "esearch + esummary 标题 + efetch 摘要；0.4s/请求限速，失败自动重试）。",
        f"- **Direct 档**（保守正则）: 任一摘要中两药名（或其查询词）出现在同一句/±150 字符"
        f"窗口内，且窗口含交互/不良反应语境词 {DIRECT_KW}。",
        "- **Class-level 建议档**: 摘要含至少一个药名，且命中该事件机制词表"
        "（按事件文本选词：血管舒张 vasorelax*/vasodilat*/hypotens*/blood pressure；"
        "吸收 absorb*/bioavailab；代谢 metaboli*；浓度 serum/plasma concentration；"
        "CNS 抑制 sedat*/CNS depress* 等）；输出命中句片段供人工裁决。",
        "- 其余 = **Not identified**。**自动结果仅为人工复核材料；最终 Evidence 列由作者"
        "裁决（R29）。FAERS 不构成独立证据（R12），dataset2 亦无 FAERS 数据。**",
        "- 注: `rank` = 该候选在其事件内的排名（每事件恰取 1 个最优 → 恒为 1）；"
        "行序按 r 全局降序。",
        "",
    ]
    by_event = {e["event"]: e for e in evidence}
    for c in cands.itertuples(index=False):
        ev = by_event[c.event]
        lines += [
            f"## {ev['a_term']} + {ev['b_term']}（{c.event}）",
            f"- 药对: `{c.drug_a}` / `{c.drug_b}`（DrugBank id）; "
            f"a_name={ev['a_name'] or '(未解析)'}, b_name={ev['b_name'] or '(未解析)'}",
            f"- 事件: {c.event}",
            f"- 模型输出（5 种子均值）: prob_mean={c.prob_mean:.4f}, "
            f"u_mean={c.u_mean:.4f}, r={c.r:.4f}",
            f"- semantic_overlap: {c.semantic_overlap}"
            + ("（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）" if c.semantic_overlap == "yes"
               else "（事件为 10 个语义不相交事件之一）"),
            f"- **evidence_auto: {_TIER_LABEL[ev['tier']]}** — {_tier_rationale(ev)}",
            "- PubMed 检索结果（retmax=5）:",
        ]
        if not ev["pmids"]:
            lines.append("  - （无检索结果）")
        else:
            for pmid in ev["pmids"]:
                lines.append(f"  - [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) "
                             f"— {ev['titles'].get(pmid, '')}")
        if ev["tier"] == "direct":
            lines.append(f"  - Direct 证据片段（PMID {ev['detail']['pmid']}）: "
                         f"“{ev['detail']['fragment'][:400]}”")
        elif ev["tier"] == "class_suggested":
            for i, frag in enumerate(ev["detail"]["fragments"], 1):
                lines.append(f"  - Class-level 命中句 {i}/2（PMID "
                             f"{ev['detail']['pmid']}，机制词 {ev['detail']['keywords']}）: "
                             f"“{frag[:400]}”")
        else:
            lines.append("  - （未检出 direct / class-level 证据）")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Task 16: dataset2 case study, per-event best candidate "
                    "+ three-tier evidence (R28/R29)")
    ap.add_argument("--csv", default=PRED_CSV)
    ap.add_argument("--fetch", default="pubmed", choices=("pubmed", "offline"))
    ap.add_argument("--retmax", type=int, default=RETMAX)
    ap.add_argument("--sleep", type=float, default=RATE_LIMIT_S)
    args = ap.parse_args(argv)

    overlap_pairs, n_overlap = cs.load_excluded_pairs()
    disjoint_events = set(json.load(open(DISJOINT_JSON, encoding="utf-8")))
    test2_events = sorted(json.load(open(TEST2_TASKS, encoding="utf-8")))
    missing_mech = set(test2_events) - set(MECH_KW)
    if missing_mech:
        raise RuntimeError(f"MECH_KW missing events: {sorted(missing_mech)}")
    print(f"[case_study_per_event] excluded dataset1-overlap pairs: {n_overlap}")
    print(f"[case_study_per_event] disjoint events (cos<0.7): {len(disjoint_events)}; "
          f"test2 events: {len(test2_events)}")

    pred = cs.load_predictions(args.csv, TIER)
    best = select_per_event(pred, overlap_pairs, test2_events)
    resolve = lru_cache(maxsize=None)(cs.resolve_names)

    rows, evidence = [], []
    for r in best.itertuples(index=False):
        a_term, b_term, a_name, b_name, a_ik, b_ik = resolve(r.drug_a, r.drug_b)
        rec = ({"pmids": [], "titles": {}, "abstracts": {}} if args.fetch == "offline"
               else pubmed_retrieve(a_term, b_term, retmax=args.retmax,
                                    sleep=args.sleep))
        tier, detail = evaluate_evidence(rec["abstracts"], a_term, b_term,
                                         MECH_KW[r.event])
        sem_ov = "no" if r.event in disjoint_events else "yes"
        rows.append({
            "rank": 1, "event": r.event, "drug_a": r.drug_a, "drug_b": r.drug_b,
            "a_name": a_name, "b_name": b_name,
            "prob_mean": r.prob_mean, "u_mean": r.u_mean, "r": r.r,
            "semantic_overlap": sem_ov, "evidence_auto": tier,
        })
        evidence.append({
            "event": r.event, "drug_a": r.drug_a, "drug_b": r.drug_b,
            "a_term": a_term, "b_term": b_term, "a_name": a_name, "b_name": b_name,
            "prob_mean": r.prob_mean, "u_mean": r.u_mean, "r": r.r,
            "semantic_overlap": sem_ov, "tier": tier, "detail": detail,
            **rec,
        })

    cands = pd.DataFrame(rows, columns=CAND_COLUMNS)
    md = render_md(cands, evidence, args.csv, n_overlap, disjoint_events,
                   test2_events)
    os.makedirs(OUTDIR, exist_ok=True)
    cands.to_csv(CAND_CSV, index=False)
    with open(EVID_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # Verification (R28): 25 rows, exactly one per event, no picking.
    assert len(cands) == N_EVENTS, f"expected {N_EVENTS} rows, got {len(cands)}"
    assert len(set(cands["event"])) == N_EVENTS, "duplicate events in candidates"
    assert set(cands["event"]) == set(test2_events), "event coverage mismatch"
    assert (cands["rank"] == 1).all(), "rank must be within-event rank (1)"
    assert cands["semantic_overlap"].isin({"yes", "no"}).all()
    assert cands["evidence_auto"].isin({"direct", "class_suggested", "none"}).all()
    assert cands[["drug_a", "drug_b", "event"]].notna().all().all()

    n_direct = sum(1 for e in evidence if e["tier"] == "direct")
    n_class = sum(1 for e in evidence if e["tier"] == "class_suggested")
    n_none = len(evidence) - n_direct - n_class
    print(f"[case_study_per_event] candidates written: {CAND_CSV}")
    print(f"[case_study_per_event] evidence written:   {EVID_MD}")
    print(f"[case_study_per_event] three-tier (R29, review material): "
          f"direct {n_direct} / class_suggested {n_class} / none {n_none} "
          f"({len(evidence)} candidates)")
    print(f"[case_study_per_event] semantic_overlap=no: "
          f"{sum(1 for e in evidence if e['semantic_overlap'] == 'no')}/{len(evidence)}")
    print(cands[["event", "drug_a", "drug_b", "a_name", "b_name",
                 "prob_mean", "u_mean", "r", "semantic_overlap",
                 "evidence_auto"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
