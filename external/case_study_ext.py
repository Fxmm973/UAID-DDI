#!/usr/bin/env python
# coding=utf-8
"""Task 10: 案例研究（独立文献佐证）— objective case-study selection with PubMed/FAERS evidence.

Selection pipeline (controller rulings R12 / R15-ext):
1. Load the 1-shot tail-corrupted variant predictions CSV (Task 8 output,
   `outputs/predictions_rxpairevid_eviddie_1shot_0shot.csv`, rows with tier=1shot),
   keep only y_true=1 candidates.
2. Aggregate per (drug_a, drug_b, event_type) the mean prob and mean uncertainty
   across the 5 train seeds; compute r = p_mean * (1 - u_mean).
3. Exclude the 304 signal pairs that overlap Dataset 1 at pair level (R15-ext).
   The build report only carries the count, so the excluded set is recomputed here
   with the same logic as build_dataset_ext R13b: signal pairs (IK14) are mapped to
   dataset1 DB ids and checked against the dataset1 train/dev/test/test2 task pairs.
   The recomputed count is asserted against the report's pair_overlap_with_dataset1.
4. Rank by r descending and take the top-10 (default).

Evidence (R12: FAERS stats do NOT count as independent evidence — labels derive from
FAERS; corroboration must come from PubMed literature):
- Per candidate, PubMed Entrez esearch with
  `"{a_name}"[All Fields] AND "{b_name}"[All Fields] AND (interaction OR adverse)`
  (retmax=3), then esummary for titles. Rate-limited 0.4 s per request (no API key).
- Heuristic corroboration flag: >=1 retrieved title mentions BOTH drug names and a
  context keyword (interact/advers/drug-drug/ddi/combination/concomitant/toxicity/
  overdose/side effect). This is a REVIEW AID ONLY, not a scientific verdict — the
  final case-study call is the user's (Task 10 Step 5). The heuristic corroboration
  count is reported in stdout and in the evidence-md header (R12 target >=7/10).

Outputs:
- outputs/case_candidates.csv — rank, drug_a, drug_b, a_name, b_name, event,
  prob_mean, u_mean, r, faers_prr_max_strict, faers_ror95_lcl_max_strict,
  n_faers_reports (names + FAERS stats joined from raw/ddi_pairs_50k.csv via IK14).
- outputs/case_evidence.md — per-candidate evidence material for human review.

CSV header adaptation: the Task 8 predictions CSV has columns run_id, train_seed,
eval_seed, setting, tier, shot, method, event_type, drug_a, drug_b, y_true, y_pred,
prob, uncertainty, evidence_0, evidence_1, checkpoint_sha256, eval_manifest_sha256,
event_embedding_sha256, git_commit. The 1-shot variant is identified by the `tier`
column (value "1shot"); `setting` holds the rare-tier marker (all rows are rare);
drug_a/drug_b are InChIKey-14 identifiers (names come from ddi_pairs_50k.csv).
"""
import argparse
import datetime
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(OUT, ".."))
DS1 = os.path.join(REPO, "PharDDIE", "dataset1")
RAW = os.path.join(OUT, "raw")
OUTDIR = os.path.join(OUT, "outputs")
PRED_CSV = os.path.join(OUTDIR, "predictions_rxpairevid_eviddie_1shot_0shot.csv")
DDI_CSV = os.path.join(RAW, "ddi_pairs_50k.csv")
BUILD_REPORT = os.path.join(OUTDIR, "dataset_ext_build_report.json")
CAND_CSV = os.path.join(OUTDIR, "case_candidates.csv")
EVID_MD = os.path.join(OUTDIR, "case_evidence.md")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UA = {"User-Agent": "PharDDIE-ext-validation/1.0"}
RATE_LIMIT_S = 0.4  # polite NCBI request spacing without an API key

# Canonical predictions-CSV header (Task 8 export; adapted: the 1-shot variant is
# identified by `tier`, drug_a/drug_b are InChIKey-14, names come from ddi_pairs_50k).
HEADER = ["run_id", "train_seed", "eval_seed", "setting", "tier", "shot", "method",
          "event_type", "drug_a", "drug_b", "y_true", "y_pred", "prob", "uncertainty",
          "evidence_0", "evidence_1", "checkpoint_sha256", "eval_manifest_sha256",
          "event_embedding_sha256", "git_commit"]

CAND_COLUMNS = ["rank", "drug_a", "drug_b", "a_name", "b_name", "event",
                "prob_mean", "u_mean", "r", "faers_prr_max_strict",
                "faers_ror95_lcl_max_strict", "n_faers_reports"]

# Context keywords for the corroboration heuristic (title must mention both drugs
# AND one of these to be flagged; deliberately permissive, human review decides).
CONTEXT_KW = ("interact", "advers", "drug-drug", "ddi", "combination",
              "concomitant", "toxicity", "overdose", "side effect")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def load_predictions(path=PRED_CSV, tier="1shot"):
    """Load the predictions CSV and keep y_true=1 rows of the given tier."""
    df = pd.read_csv(path)
    df = df[(df["tier"] == tier) & (df["y_true"] == 1)].copy()
    if df.empty:
        raise ValueError(f"no y_true=1 rows for tier={tier!r} in {path}")
    return df


def aggregate_candidates(df):
    """Per (drug_a, drug_b, event_type): mean prob / mean uncertainty over seeds,
    then r = p_mean * (1 - u_mean)."""
    g = df.groupby(["drug_a", "drug_b", "event_type"], as_index=False)[
        ["prob", "uncertainty"]].mean()
    g = g.rename(columns={"prob": "prob_mean", "uncertainty": "u_mean",
                          "event_type": "event"})
    g["r"] = g["prob_mean"] * (1.0 - g["u_mean"])
    return g


def load_overlap_pairs(ddi_csv=DDI_CSV, ds1_dir=DS1):
    """Recompute the R15-ext excluded set: signal pairs (IK14) that overlap Dataset 1
    at pair level. Replicates build_dataset_ext R13b so the count must equal the
    build report's pair_overlap_with_dataset1 (304)."""
    from build_dataset_ext import build_ik14_to_db_ids, dataset1_task_pairs
    df = pd.read_csv(ddi_csv, dtype={"drug_a_ik14": "string", "drug_b_ik14": "string"})
    sig = df[df["faers_ror95_lcl_max_strict"].notnull()]
    sig_pairs = {tuple(sorted((r["drug_a_ik14"], r["drug_b_ik14"])))
                 for _, r in sig.iterrows()}
    ds1_smiles = pd.read_csv(os.path.join(ds1_dir, "drug_smiles.csv"),
                             dtype={"drug_id": "string"})
    ik14_to_db_ids = build_ik14_to_db_ids(ds1_smiles)
    ds1_pairs = dataset1_task_pairs(ds1_dir)
    overlap = set()
    for a, b in sig_pairs:
        da, db = ik14_to_db_ids.get(a), ik14_to_db_ids.get(b)
        if not da or not db:
            continue
        if any(tuple(sorted((x, y))) in ds1_pairs for x in da for y in db):
            overlap.add((a, b))
    return overlap


def rank_candidates(agg, overlap_pairs, top_n=10):
    """Drop pairs in overlap_pairs (sorted IK14 tuples), rank by r desc, take top_n."""
    mask = pd.Series([tuple(sorted((a, b))) in overlap_pairs
                      for a, b in zip(agg["drug_a"], agg["drug_b"])], index=agg.index)
    cand = agg.loc[~mask].sort_values("r", ascending=False).head(top_n).copy()
    cand["rank"] = range(1, len(cand) + 1)
    return cand[["rank", "drug_a", "drug_b", "event", "prob_mean", "u_mean", "r"]].reset_index(drop=True)


def join_faers(cand, ddi_df=None):
    """Join a_name/b_name and FAERS signal stats from ddi_pairs_50k.csv via IK14
    (order-independent: keyed on the sorted pair)."""
    if ddi_df is None:
        ddi_df = pd.read_csv(DDI_CSV, dtype={"drug_a_ik14": "string", "drug_b_ik14": "string"})
    lookup = {}
    for r in ddi_df.itertuples(index=False):
        lookup[tuple(sorted((r.drug_a_ik14, r.drug_b_ik14)))] = r
    rows = []
    for r in cand.itertuples(index=False):
        rec = lookup.get(tuple(sorted((r.drug_a, r.drug_b))))
        n_reports = None
        if rec is not None and pd.notnull(rec.n_faers_reports):
            n_reports = int(rec.n_faers_reports)
        rows.append({
            "rank": r.rank, "drug_a": r.drug_a, "drug_b": r.drug_b,
            "a_name": rec.a_name if rec is not None else "",
            "b_name": rec.b_name if rec is not None else "",
            "event": r.event, "prob_mean": r.prob_mean, "u_mean": r.u_mean, "r": r.r,
            "faers_prr_max_strict": rec.faers_prr_max_strict if rec is not None else None,
            "faers_ror95_lcl_max_strict": rec.faers_ror95_lcl_max_strict if rec is not None else None,
            "n_faers_reports": n_reports,
        })
    return pd.DataFrame(rows, columns=CAND_COLUMNS)


# ---------------------------------------------------------------------------
# PubMed evidence (Entrez esearch + esummary; rate-limited, no API key)
# ---------------------------------------------------------------------------

def build_esearch_term(a_name, b_name):
    return (f'"{a_name}"[All Fields] AND "{b_name}"[All Fields] '
            f"AND (interaction OR adverse)")


def _eutils_get(params, endpoint="esearch.fcgi"):
    url = f"{EUTILS}/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def parse_esearch_ids(text):
    """esearch JSON -> list of PMIDs (already capped by retmax upstream)."""
    return json.loads(text)["esearchresult"]["idlist"]


def parse_esummary_titles(text):
    """esummary JSON -> {pmid: title} (entries without a title are skipped)."""
    d = json.loads(text)
    res = d.get("result", {})
    out = {}
    for pmid in res.get("uids", []):
        rec = res.get(pmid) or {}
        if isinstance(rec, dict) and rec.get("title"):
            out[pmid] = rec["title"]
    return out


def pubmed_fetch(a_name, b_name, retmax=3, sleep=RATE_LIMIT_S):
    """esearch top-`retmax` PMIDs + esummary titles for a candidate pair.
    Returns [(pmid, title), ...]; [] when nothing is retrieved."""
    term = build_esearch_term(a_name, b_name)
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


def corroborated(title, a_name, b_name):
    """Heuristic: title mentions BOTH drug names and a context keyword. Review aid
    only — NOT a scientific verdict (final call is the user's)."""
    t = title.lower()
    if a_name.lower() not in t or b_name.lower() not in t:
        return False
    return any(kw in t for kw in CONTEXT_KW)


def gather_evidence(cands, fetch=pubmed_fetch):
    """Per candidate: top-3 (pmid, title) hits + heuristic corroboration flag.
    `fetch` is injectable for tests (no network in tests)."""
    out = []
    for r in cands.itertuples(index=False):
        hits = fetch(r.a_name, r.b_name)
        out.append({
            "rank": r.rank, "drug_a": r.drug_a, "drug_b": r.drug_b,
            "a_name": r.a_name, "b_name": r.b_name, "event": r.event,
            "prob_mean": r.prob_mean, "u_mean": r.u_mean, "r": r.r,
            "faers_prr_max_strict": r.faers_prr_max_strict,
            "faers_ror95_lcl_max_strict": r.faers_ror95_lcl_max_strict,
            "n_faers_reports": r.n_faers_reports,
            "pmids": [p for p, _ in hits],
            "titles": [t for _, t in hits],
            "corroborated": any(corroborated(t, r.a_name, r.b_name) for _, t in hits),
        })
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_evidence_md(cands, evidence, src_csv, n_overlap):
    n = len(evidence)
    n_cor = sum(1 for e in evidence if e["corroborated"])
    lines = [
        "# Case-Study Evidence: PubMed 独立文献佐证（人工复核材料）",
        "",
        f"- 生成日期: {datetime.date.today().isoformat()}",
        f"- 来源预测 CSV: `{os.path.basename(src_csv)}`（1-shot tier, y_true=1）",
        f"- 选择规则: 每 (drug_a, drug_b, event) 在 5 个训练种子上取 prob/uncertainty 均值；"
        f"r = p_mean·(1−u_mean)；剔除与 Dataset 1 药对级重叠的 {n_overlap} 个信号对 (R15-ext)；"
        "按 r 降序取 top-10。",
        f"- **启发式佐证计数: {n_cor}/{n}**（标题同时提及两药且含交互/不良反应语境关键词；"
        "仅作人工复核提示，非最终科学判断；R12 目标 ≥7/10；FAERS 信号统计不视为独立证据）",
        f"- 检索式: `\"a_name\"[All Fields] AND \"b_name\"[All Fields] AND (interaction OR adverse)`，"
        "PubMed Entrez，top-3 PMID+标题，0.4s/请求限速。",
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
            f"## Rank {r.rank} — {r.a_name} + {r.b_name}（{r.event}）",
            f"- 药对: `{r.drug_a}` / `{r.drug_b}`（IK14）",
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
    ap = argparse.ArgumentParser(description="Task 10: case-study selection + PubMed/FAERS evidence")
    ap.add_argument("--csv", default=PRED_CSV, help="predictions CSV (default: 1shot/0shot export)")
    ap.add_argument("--tier", default="1shot", help="tier column value to select (default: 1shot)")
    ap.add_argument("--top-n", type=int, default=10, help="number of candidates (default: 10)")
    ap.add_argument("--fetch", default="pubmed", choices=("pubmed", "offline"),
                    help="fetch=offline skips the network and records no hits")
    args = ap.parse_args(argv)

    overlap = load_overlap_pairs()
    report = json.load(open(BUILD_REPORT))
    expected = report["pair_overlap_with_dataset1"]
    if len(overlap) != expected:
        raise RuntimeError(
            f"recomputed overlap set size {len(overlap)} != build report "
            f"pair_overlap_with_dataset1={expected} (R15-ext invariant broken)")
    print(f"[case_study_ext] overlap pairs excluded: {len(overlap)} "
          f"(== build report pair_overlap_with_dataset1={expected})")

    pred = load_predictions(args.csv, args.tier)
    agg = aggregate_candidates(pred)
    cands = join_faers(rank_candidates(agg, overlap, args.top_n))
    if args.fetch == "offline":
        evidence = gather_evidence(cands, fetch=lambda a, b: [])
    else:
        evidence = gather_evidence(cands)  # real PubMed Entrez, 0.4 s/request
    md = render_evidence_md(cands, evidence, args.csv, len(overlap))
    os.makedirs(OUTDIR, exist_ok=True)
    cands.to_csv(CAND_CSV, index=False)
    with open(EVID_MD, "w", encoding="utf-8") as f:
        f.write(md)

    n_cor = sum(1 for e in evidence if e["corroborated"])
    print(f"[case_study_ext] candidates written: {CAND_CSV}")
    print(f"[case_study_ext] evidence written:   {EVID_MD}")
    print(f"[case_study_ext] heuristic corroboration: {n_cor}/{len(evidence)} "
          f"(R12 target >=7/10; human review required)")
    print(cands[["rank", "a_name", "b_name", "event", "prob_mean", "u_mean", "r"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
