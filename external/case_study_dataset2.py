#!/usr/bin/env python
# coding=utf-8
"""Task 12 (Step 6): dataset2 case study — objective selection + PubMed evidence.

Selection pipeline (mirrors the reviewed case_study_ext.py for the dataset2
cross-benchmark tier):
1. Load predictions_dataset2_eviddie_0shot.csv, keep y_true=1 rows of
   tier='test2'.
2. Aggregate per (drug_a, drug_b, event_type) the mean prob / mean uncertainty
   over the 5 train seeds; r = p_mean * (1 - u_mean).
3. Exclude candidates whose unordered pair overlaps Dataset 1 (loaded from
   external/outputs/dataset2_pair_overlap.json, computed by
   audit_overlap_dataset2.py over the four dataset1 task files).
4. Rank by r descending, take the top-10.

Names (controller ruling b — dataset2 has no name file):
  DB id -> SMILES (dataset1 drug_smiles.csv, else dataset2 drug_smiles.csv)
        -> InChIKey-14 -> RxPairEvid name table (raw/ddi_pairs_50k.csv
           a_name/b_name by IK14). Unresolved ids fall back to the PubMed
           query term "DrugBank {DB_ID}".

Evidence (R12): PubMed Entrez esearch + esummary, 0.4 s/request rate limit,
heuristic corroboration = a title mentioning both query terms and a context
keyword. Review aid only — the final case-study call is the user's.

Outputs: external/outputs/case_candidates_dataset2.csv
         external/outputs/case_evidence_dataset2.md
"""
import argparse
import datetime
import json
import os
import time

import pandas as pd

from case_study_ext import (CONTEXT_KW, _eutils_get, build_esearch_term,
                            corroborated, parse_esearch_ids,
                            parse_esummary_titles)

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(OUT, ".."))
DS1 = os.path.join(REPO, "PharDDIE", "dataset1")
DS2 = os.path.join(REPO, "EviDDIE", "dataset2")
RAW = os.path.join(OUT, "raw")
OUTDIR = os.path.join(OUT, "outputs")
PRED_CSV = os.path.join(OUTDIR, "predictions_dataset2_eviddie_0shot.csv")
DDI_CSV = os.path.join(RAW, "ddi_pairs_50k.csv")
PAIR_OVERLAP_JSON = os.path.join(OUTDIR, "dataset2_pair_overlap.json")
DRUG_OVERLAP_JSON = os.path.join(OUTDIR, "dataset2_drug_overlap.json")
CAND_CSV = os.path.join(OUTDIR, "case_candidates_dataset2.csv")
EVID_MD = os.path.join(OUTDIR, "case_evidence_dataset2.md")
TIER = "test2"

CAND_COLUMNS = ["rank", "drug_a", "drug_b", "a_name", "b_name", "event",
                "prob_mean", "u_mean", "r", "faers_prr_max_strict",
                "faers_ror95_lcl_max_strict", "n_faers_reports"]


# ---------------------------------------------------------------------------
# Names: DB id -> IK14 -> RxPairEvid name table
# ---------------------------------------------------------------------------

def load_ik14_by_db():
    """DB id -> IK14, preferring dataset1's smiles, falling back to dataset2's."""
    from audit_overlap_ext import ik14_of
    out = {}
    for csv_path in (os.path.join(DS1, "drug_smiles.csv"),
                     os.path.join(DS2, "drug_smiles.csv")):
        df = pd.read_csv(csv_path, dtype={"drug_id": "string"})
        for _, r in df.iterrows():
            if r["drug_id"] in out:
                continue
            ik = ik14_of(r["smiles"])
            if ik:
                out[r["drug_id"]] = ik
    return out


def load_name_table():
    """IK14 -> name (a_name/b_name from ddi_pairs_50k.csv)."""
    df = pd.read_csv(DDI_CSV, dtype={"drug_a_ik14": "string", "drug_b_ik14": "string"})
    names = {}
    for _, r in df.iterrows():
        if pd.notnull(r["drug_a_ik14"]) and pd.notnull(r["a_name"]):
            names.setdefault(r["drug_a_ik14"], r["a_name"])
        if pd.notnull(r["drug_b_ik14"]) and pd.notnull(r["b_name"]):
            names.setdefault(r["drug_b_ik14"], r["b_name"])
    return names


def resolve_names(db_a, db_b):
    """Return (a_query_term, b_query_term, a_name, b_name, a_ik14, b_ik14).

    a_name/b_name: resolved names ('' when unresolved); query terms fall back
    to 'DrugBank {DB_ID}' for unresolved ids (controller ruling b)."""
    ik14_by_db = load_ik14_by_db()
    name_table = load_name_table()
    a_ik, b_ik = ik14_by_db.get(db_a), ik14_by_db.get(db_b)
    a_name = name_table.get(a_ik, "") if a_ik else ""
    b_name = name_table.get(b_ik, "") if b_ik else ""
    a_term = a_name if a_name else f"DrugBank {db_a}"
    b_term = b_name if b_name else f"DrugBank {db_b}"
    return a_term, b_term, a_name, b_name, a_ik or "", b_ik or ""


def load_faers_lookup():
    df = pd.read_csv(DDI_CSV, dtype={"drug_a_ik14": "string", "drug_b_ik14": "string"})
    lookup = {}
    for r in df.itertuples(index=False):
        lookup[tuple(sorted((r.drug_a_ik14, r.drug_b_ik14)))] = r
    return lookup


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def load_predictions(path=PRED_CSV, tier=TIER):
    df = pd.read_csv(path)
    df = df[(df["tier"] == tier) & (df["y_true"] == 1)].copy()
    if df.empty:
        raise ValueError(f"no y_true=1 rows for tier={tier!r} in {path}")
    return df


def aggregate_candidates(df):
    g = df.groupby(["drug_a", "drug_b", "event_type"], as_index=False)[
        ["prob", "uncertainty"]].mean()
    g = g.rename(columns={"prob": "prob_mean", "uncertainty": "u_mean",
                          "event_type": "event"})
    g["r"] = g["prob_mean"] * (1.0 - g["u_mean"])
    return g


def load_excluded_pairs():
    """Overlap with dataset1 at pair level (from audit_overlap_dataset2, step 4b)."""
    rec = json.load(open(PAIR_OVERLAP_JSON, encoding="utf-8"))
    return {tuple(p) for p in rec["overlap_pairs"]}, rec["n_overlap_pairs"]


def rank_candidates(agg, overlap_pairs, top_n=10):
    mask = pd.Series([tuple(sorted((a, b))) in overlap_pairs
                      for a, b in zip(agg["drug_a"], agg["drug_b"])], index=agg.index)
    cand = agg.loc[~mask].sort_values("r", ascending=False).head(top_n).copy()
    cand["rank"] = range(1, len(cand) + 1)
    return cand.reset_index(drop=True)


def join_names_faers(cand, faers_lookup):
    rows = []
    for r in cand.itertuples(index=False):
        a_term, b_term, a_name, b_name, a_ik, b_ik = resolve_names(r.drug_a, r.drug_b)
        rec = None
        if a_ik and b_ik:
            rec = faers_lookup.get(tuple(sorted((a_ik, b_ik))))
        n_reports = None
        if rec is not None and pd.notnull(rec.n_faers_reports):
            n_reports = int(rec.n_faers_reports)
        rows.append({
            "rank": r.rank, "drug_a": r.drug_a, "drug_b": r.drug_b,
            "a_name": a_name, "b_name": b_name,
            "event": r.event, "prob_mean": r.prob_mean, "u_mean": r.u_mean, "r": r.r,
            "faers_prr_max_strict": rec.faers_prr_max_strict if rec is not None else None,
            "faers_ror95_lcl_max_strict": rec.faers_ror95_lcl_max_strict if rec is not None else None,
            "n_faers_reports": n_reports,
        })
    return pd.DataFrame(rows, columns=CAND_COLUMNS)


# ---------------------------------------------------------------------------
# PubMed evidence (reuses case_study_ext network helpers; event text is
# appended to the term only when a name is unresolved, per controller ruling b)
# ---------------------------------------------------------------------------

def build_term(a_term, b_term, event):
    if "DrugBank " in a_term or "DrugBank " in b_term:
        return (f'"{a_term}"[All Fields] AND "{b_term}"[All Fields] '
                f'AND "{event}"[All Fields] AND (interaction OR adverse)')
    return build_esearch_term(a_term, b_term)


def pubmed_fetch_terms(a_term, b_term, event, retmax=3, sleep=0.4):
    term = build_term(a_term, b_term, event)
    body = _eutils_get({"db": "pubmed", "retmode": "json", "retmax": retmax,
                        "term": term, "tool": "PharDDIE-ext-validation"})
    time.sleep(sleep)
    ids = parse_esearch_ids(body)
    if not ids:
        return []
    body2 = _eutils_get({"db": "pubmed", "retmode": "json", "id": ",".join(ids),
                         "tool": "PharDDIE-ext-validation"}, endpoint="esummary.fcgi")
    time.sleep(sleep)
    titles = parse_esummary_titles(body2)
    return [(pmid, titles.get(pmid, "")) for pmid in ids]


def gather_evidence(cands, fetch=pubmed_fetch_terms):
    out = []
    for r in cands.itertuples(index=False):
        a_term, b_term, _, _, _, _ = resolve_names(r.drug_a, r.drug_b)
        hits = fetch(a_term, b_term, r.event)
        out.append({
            "rank": r.rank, "drug_a": r.drug_a, "drug_b": r.drug_b,
            "a_name": r.a_name, "b_name": r.b_name, "event": r.event,
            "a_query_term": a_term, "b_query_term": b_term,
            "prob_mean": r.prob_mean, "u_mean": r.u_mean, "r": r.r,
            "faers_prr_max_strict": r.faers_prr_max_strict,
            "faers_ror95_lcl_max_strict": r.faers_ror95_lcl_max_strict,
            "n_faers_reports": r.n_faers_reports,
            "pmids": [p for p, _ in hits],
            "titles": [t for _, t in hits],
            "corroborated": any(corroborated(t, a_term, b_term) for _, t in hits),
        })
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_evidence_md(cands, evidence, src_csv, n_overlap):
    n = len(evidence)
    n_cor = sum(1 for e in evidence if e["corroborated"])
    lines = [
        "# Case-Study Evidence (Dataset 2): PubMed 独立文献佐证（人工复核材料）",
        "",
        f"- 生成日期: {datetime.date.today().isoformat()}",
        f"- 来源预测 CSV: `{os.path.basename(src_csv)}`（tier=test2, y_true=1, 5 种子）",
        f"- 选择规则: 每 (drug_a, drug_b, event) 在 5 个训练种子上取 prob/uncertainty 均值；"
        f"r = p_mean·(1−u_mean)；剔除与 Dataset 1 药对级重叠的 {n_overlap} 个 test2 药对"
        "（dataset2_pair_overlap.json）；按 r 降序取 top-10。",
        f"- 药名解析: DB ID → drug_smiles → InChIKey-14 → RxPairEvid 名字表"
        f"（ddi_pairs_50k.csv）；未解析的药对用 `DrugBank {{DB_ID}}` 作 PubMed 查询词"
        "（dataset2 无名字文件，controller ruling b）。",
        f"- **启发式佐证计数: {n_cor}/{n}**（标题同时提及两药查询词且含交互/不良反应语境关键词；"
        "仅作人工复核提示，非最终科学判断；FAERS 信号统计不视为独立证据）",
        f"- 检索式: `\"a_name\"[All Fields] AND \"b_name\"[All Fields] AND (interaction OR adverse)`"
        "（未解析名字时追加事件文本关键词），PubMed Entrez，top-3 PMID+标题，0.4s/请求限速。",
        "",
        "## 证据列说明（论文表格用）",
        "- 独立证据列：PubMed PMID（唯一独立佐证来源，R12）",
        "- 非独立证据列：FAERS 统计（标签源自 FAERS，仅作背景摘录，不计入佐证计数）",
        "",
    ]
    by_rank = {e["rank"]: e for e in evidence}
    for r in cands.itertuples(index=False):
        ev = by_rank.get(r.rank)
        flag = "YES (启发式, 需人工复核)" if ev and ev["corroborated"] else "NO"
        lines += [
            f"## Rank {r.rank} — {ev['a_query_term']} + {ev['b_query_term']}（{r.event}）",
            f"- 药对: `{r.drug_a}` / `{r.drug_b}`（DrugBank id）",
            f"- 模型输出（5 种子均值）: prob_mean={r.prob_mean:.4f}, "
            f"u_mean={r.u_mean:.4f}, r={r.r:.4f}",
            f"- FAERS（非独立证据，仅摘录）: n_reports={r.n_faers_reports}, "
            f"PRR_max_strict={r.faers_prr_max_strict}, "
            f"ROR95_lcl_max_strict={r.faers_ror95_lcl_max_strict}",
            f"- 佐证标志: {flag}",
            "- PubMed 检索结果:",
        ]
        if not ev or not ev["pmids"]:
            lines.append("  - （无检索结果）")
        else:
            for i, (pmid, title) in enumerate(zip(ev["pmids"], ev["titles"]), 1):
                lines.append(f"  {i}. [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) — {title}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Task 12: dataset2 case-study selection + PubMed evidence")
    ap.add_argument("--csv", default=PRED_CSV)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--fetch", default="pubmed", choices=("pubmed", "offline"))
    args = ap.parse_args(argv)

    overlap_pairs, n_overlap = load_excluded_pairs()
    drug_overlap = json.load(open(DRUG_OVERLAP_JSON, encoding="utf-8"))
    print(f"[case_study_dataset2] excluded dataset1-overlap pairs: {n_overlap}")

    pred = load_predictions(args.csv, TIER)
    agg = aggregate_candidates(pred)
    cands = join_names_faers(rank_candidates(agg, overlap_pairs, args.top_n),
                             load_faers_lookup())
    if args.fetch == "offline":
        evidence = gather_evidence(cands, fetch=lambda a, b, e: [])
    else:
        evidence = gather_evidence(cands)
    md = render_evidence_md(cands, evidence, args.csv, n_overlap)
    os.makedirs(OUTDIR, exist_ok=True)
    cands.to_csv(CAND_CSV, index=False)
    with open(EVID_MD, "w", encoding="utf-8") as f:
        f.write(md)

    n_cor = sum(1 for e in evidence if e["corroborated"])
    print(f"[case_study_dataset2] candidates written: {CAND_CSV}")
    print(f"[case_study_dataset2] evidence written:   {EVID_MD}")
    print(f"[case_study_dataset2] heuristic corroboration: {n_cor}/{len(evidence)} "
          f"(R12 target >=7/10; human review required)")
    print(cands[["rank", "drug_a", "drug_b", "a_name", "b_name", "event",
                 "prob_mean", "u_mean", "r"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
