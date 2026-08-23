#!/usr/bin/env python
# coding=utf-8
"""Task 15: dataset2 case-study FINAL — top-20 objective selection + PubMed evidence.

Thin adaptation of case_study_dataset2.py (Task 12, reviewed) applying the
final controller rulings for the external-validation deliverable:

- R24: pre-registered top-20; ALL 20 candidates are reported — no picking, no
  dropping. Uncorroborated candidates stay in the table and are framed as
  triage/referral candidates for human review.
- R26: per-candidate `semantic_overlap` flag: "yes" if the candidate's event
  has a Dataset-1 counterpart with cosine >= 0.7 (i.e. the event is NOT in
  external/outputs/disjoint_events.json), "no" otherwise.
- R12: corroboration = PubMed literature mentioning both drugs in an
  interaction/adverse context; the title-heuristic (both query terms + a
  context keyword) is a REVIEW AID only, not a scientific verdict. FAERS
  statistics do not count as corroboration, and there are none for dataset2
  (the FAERS columns are kept empty in the final CSV).

Inputs:
  external/outputs/predictions_ds2_retrained_0shot.csv — 18,720 rows,
    tier=test2, y_true=1/0, 5 train seeds (19940419, 20230801, 20240520,
    20260201, 20260301), retrained on Dataset 2.
  external/outputs/disjoint_events.json — the 10 Dataset-2 test2 events whose
    max cosine to any Dataset-1 event is < 0.7 (semantic disjoint).
  external/outputs/dataset2_pair_overlap.json — pair-level Dataset-1 overlap
    (computed by audit_overlap_dataset2.py; exclusion reused as-is).

Reuse decisions (vs copying):
  - Selection pipeline (load_predictions, aggregate_candidates,
    load_excluded_pairs) imported from case_study_dataset2 unchanged.
  - `rank_candidates` is NOT imported: the reference sorts on r alone with
    pandas' default (unstable) quicksort; the final version adds an explicit
    (drug_a, drug_b, event) tie-break so the output is fully deterministic
    (controller determinism rule). The exclusion mask logic is identical.
  - Name resolution (`resolve_names`) imported; memoized via lru_cache because
    the reference reloads both drug_smiles tables and recomputes InChIKeys on
    every call (~1.9 s/call; 20 candidates would pay it ~3x).
  - FAERS joining is NOT reused: the spec says the ds2 FAERS columns are
    empty (no FAERS data for ds2; FAERS stats are not independent evidence,
    R12). `join_names` below fills them with "".
  - PubMed evidence (`gather_evidence`, `pubmed_fetch_terms` and the
    case_study_ext helpers) imported unchanged.
  - Markdown rendering is a new function: the final md adds the
    semantic_overlap flag, the top-20/R24 framing and drops FAERS lines.

Outputs:
  external/outputs/case_candidates_dataset2_final.csv (top-20, r-sorted)
  external/outputs/case_evidence_dataset2_final.md (PubMed evidence)
"""
import argparse
import datetime
import json
import os
from functools import lru_cache

import pandas as pd

import case_study_dataset2 as cs

OUT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(OUT, "outputs")
PRED_CSV = os.path.join(OUTDIR, "predictions_ds2_retrained_0shot.csv")
DISJOINT_JSON = os.path.join(OUTDIR, "disjoint_events.json")
CAND_CSV = os.path.join(OUTDIR, "case_candidates_dataset2_final.csv")
EVID_MD = os.path.join(OUTDIR, "case_evidence_dataset2_final.md")
TIER = "test2"
TOP_N = 20

# Final CSV columns, order per task spec (FAERS columns empty for ds2, R12).
CAND_COLUMNS = ["rank", "drug_a", "drug_b", "a_name", "b_name", "event",
                "prob_mean", "u_mean", "r", "semantic_overlap",
                "n_faers_reports", "faers_prr_max_strict",
                "faers_ror95_lcl_max_strict"]


# ---------------------------------------------------------------------------
# Deterministic ranking (reference rank_candidates + explicit tie-break)
# ---------------------------------------------------------------------------

def rank_candidates_deterministic(agg, overlap_pairs, top_n=TOP_N):
    """Same exclusion as cs.rank_candidates, but ties in r are broken by
    (drug_a, drug_b, event) so the top-N is identical on any platform."""
    mask = pd.Series([tuple(sorted((a, b))) in overlap_pairs
                      for a, b in zip(agg["drug_a"], agg["drug_b"])],
                     index=agg.index)
    cand = agg.loc[~mask].sort_values(
        ["r", "drug_a", "drug_b", "event"], ascending=False).head(top_n).copy()
    cand["rank"] = range(1, len(cand) + 1)
    return cand.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Names + semantic_overlap flag (R26); FAERS columns empty for ds2 (R12)
# ---------------------------------------------------------------------------

# resolve_names reloads both drug_smiles tables and recomputes InChIKeys on
# every call; memoize it (identical semantics, ~50x faster for 20 candidates).
_resolve_cached = lru_cache(maxsize=None)(cs.resolve_names)


def join_names(cand, disjoint_events):
    """Add a_name/b_name (via the reference IK14 resolution), the
    semantic_overlap flag (R26), and empty FAERS columns (R12)."""
    rows = []
    for r in cand.itertuples(index=False):
        _, _, a_name, b_name, _, _ = _resolve_cached(r.drug_a, r.drug_b)
        rows.append({
            "rank": r.rank, "drug_a": r.drug_a, "drug_b": r.drug_b,
            "a_name": a_name, "b_name": b_name, "event": r.event,
            "prob_mean": r.prob_mean, "u_mean": r.u_mean, "r": r.r,
            "semantic_overlap": "no" if r.event in disjoint_events else "yes",
            "n_faers_reports": "", "faers_prr_max_strict": "",
            "faers_ror95_lcl_max_strict": "",
        })
    return pd.DataFrame(rows, columns=CAND_COLUMNS)


# ---------------------------------------------------------------------------
# Evidence: reuse cs.gather_evidence / cs.pubmed_fetch_terms unchanged. The
# fetch is injectable (offline mode for dry-runs; no network).
# ---------------------------------------------------------------------------

def gather_evidence(cands, fetch=cs.pubmed_fetch_terms):
    return cs.gather_evidence(cands, fetch=fetch)


# ---------------------------------------------------------------------------
# Rendering (final md: R24/R26/R12 framing)
# ---------------------------------------------------------------------------

def render_evidence_md_final(cands, evidence, src_csv, n_overlap, disjoint_events):
    n = len(evidence)
    n_cor = sum(1 for e in evidence if e["corroborated"])
    n_disjoint = sum(1 for e in evidence if e["event"] in disjoint_events)
    lines = [
        "# Case-Study Evidence (Dataset 2, FINAL): PubMed 独立文献佐证（人工复核材料）",
        "",
        f"- 生成日期: {datetime.date.today().isoformat()}",
        f"- 来源预测 CSV: `{os.path.basename(src_csv)}`（tier=test2, y_true=1, "
        "5 训练种子: 19940419 / 20230801 / 20240520 / 20260201 / 20260301，Dataset 2 上重训）",
        f"- 选择规则: 每 (drug_a, drug_b, event) 在 5 个训练种子上取 prob/uncertainty 均值；"
        f"r = p_mean·(1−u_mean)；剔除与 Dataset 1 药对级重叠的 {n_overlap} 个 test2 药对"
        "（dataset2_pair_overlap.json）；按 r 降序取 **pre-registered top-20**（R24），"
        "**全部 20 个候选均列于此表，无剔除、无挑选**。",
        f"- 药名解析: DB ID → drug_smiles → InChIKey-14 → RxPairEvid 名字表"
        f"（ddi_pairs_50k.csv）；未解析的药对用 `DrugBank {{DB_ID}}` 作 PubMed 查询词"
        "（dataset2 无名字文件，controller ruling b）。",
        f"- **semantic_overlap 标志（R26）**: “yes” = 该候选事件与 Dataset 1 某事件 max cosine "
        f"≥ 0.7（即不在 disjoint_events.json）；“no” = 事件在 10 个语义不相交事件中 "
        f"（max cosine < 0.7）。本表 {n_disjoint} 个候选为 “no”。",
        f"- **启发式佐证计数: {n_cor}/{n}**（标题同时提及两药查询词且含交互/不良反应"
        "语境关键词；仅作人工复核提示，**非最终科学判断**；FAERS 信号统计不视为独立证据，"
        "且 dataset2 无 FAERS 数据，相关列为空）",
        "- 检索式: `\"a_name\"[All Fields] AND \"b_name\"[All Fields] AND (interaction OR adverse)`"
        "（未解析名字时追加事件文本关键词），PubMed Entrez，top-3 PMID+标题，0.4s/请求限速。",
        "- **triage/referral 框架（R24）**: 未获文献佐证的候选不剔除，仍列于表中，"
        "标注为需人工优先复核的 triage/referral 候选。最终案例结论由作者裁决。",
        "",
    ]
    by_rank = {e["rank"]: e for e in evidence}
    for r in cands.itertuples(index=False):
        ev = by_rank.get(r.rank)
        flag = "YES（启发式，需人工复核）" if ev and ev["corroborated"] else \
            "NO（未获独立文献佐证；triage/referral 候选）"
        ov = r.semantic_overlap
        lines += [
            f"## Rank {r.rank} — {ev['a_query_term']} + {ev['b_query_term']}（{r.event}）",
            f"- 药对: `{r.drug_a}` / `{r.drug_b}`（DrugBank id）; "
            f"a_name={r.a_name or '(未解析)'}, b_name={r.b_name or '(未解析)'}",
            f"- 事件: {r.event}",
            f"- 模型输出（5 种子均值）: prob_mean={r.prob_mean:.4f}, "
            f"u_mean={r.u_mean:.4f}, r={r.r:.4f}",
            f"- semantic_overlap: {ov}"
            + ("（事件与 Dataset 1 有 cosine ≥ 0.7 对应事件）" if ov == "yes"
               else "（事件为 10 个语义不相交事件之一）"),
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
    ap = argparse.ArgumentParser(description="Task 15: dataset2 case-study FINAL (top-20)")
    ap.add_argument("--csv", default=PRED_CSV)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--fetch", default="pubmed", choices=("pubmed", "offline"))
    args = ap.parse_args(argv)

    overlap_pairs, n_overlap = cs.load_excluded_pairs()
    disjoint_events = set(json.load(open(DISJOINT_JSON, encoding="utf-8")))
    print(f"[case_study_dataset2_final] excluded dataset1-overlap pairs: {n_overlap}")
    print(f"[case_study_dataset2_final] disjoint events (max cosine < 0.7): "
          f"{len(disjoint_events)}")

    pred = cs.load_predictions(args.csv, TIER)
    agg = cs.aggregate_candidates(pred)
    cands = join_names(rank_candidates_deterministic(agg, overlap_pairs, args.top_n),
                       disjoint_events)
    if args.fetch == "offline":
        evidence = gather_evidence(cands, fetch=lambda a, b, e: [])
    else:
        evidence = gather_evidence(cands)
    md = render_evidence_md_final(cands, evidence, args.csv, n_overlap,
                                  disjoint_events)
    os.makedirs(OUTDIR, exist_ok=True)
    cands.to_csv(CAND_CSV, index=False)
    with open(EVID_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # Verification: exactly 20 rows, ranks 1..20, r-sorted, all flagged.
    assert len(cands) == args.top_n, f"expected {args.top_n} candidates, got {len(cands)}"
    assert list(cands["rank"]) == list(range(1, args.top_n + 1))
    assert cands["r"].is_monotonic_decreasing, "candidates not r-sorted"
    assert cands["semantic_overlap"].isin({"yes", "no"}).all()
    assert cands[["drug_a", "drug_b", "event"]].notna().all().all()
    n_cor = sum(1 for e in evidence if e["corroborated"])
    print(f"[case_study_dataset2_final] candidates written: {CAND_CSV}")
    print(f"[case_study_dataset2_final] evidence written:   {EVID_MD}")
    print(f"[case_study_dataset2_final] heuristic corroboration (review aid only, "
          f"R12): {n_cor}/{len(evidence)}")
    print(f"[case_study_dataset2_final] semantic_overlap=no (disjoint event): "
          f"{sum(1 for e in evidence if e['event'] in disjoint_events)}/{len(evidence)}")
    print(cands[["rank", "drug_a", "drug_b", "a_name", "b_name", "event",
                 "prob_mean", "u_mean", "r", "semantic_overlap"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
