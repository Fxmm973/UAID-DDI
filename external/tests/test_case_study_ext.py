#!/usr/bin/env python
# coding=utf-8
"""Tests for external/case_study_ext.py (Task 10: case-study selection + evidence).

No network: PubMed fetching is exercised through an injected stub fetch function;
the esearch/esummary JSON parsers are tested against canned payloads.
"""
import json

import pandas as pd
import pytest

from case_study_ext import (
    HEADER,
    aggregate_candidates,
    build_esearch_term,
    corroborated,
    gather_evidence,
    join_faers,
    load_predictions,
    parse_esearch_ids,
    parse_esummary_titles,
    rank_candidates,
    render_evidence_md,
)


def _row(drug_a, drug_b, event, seed, p, u, y_true=1, tier="1shot"):
    """One predictions-CSV row with the full 20-column header (dummy extras)."""
    return dict(
        run_id="r", train_seed=seed, eval_seed=seed, setting="rare", tier=tier,
        shot=0, method="EviDDIE", event_type=event, drug_a=drug_a, drug_b=drug_b,
        y_true=y_true, y_pred=int(y_true), prob=p, uncertainty=u,
        evidence_0="", evidence_1="", checkpoint_sha256="c",
        eval_manifest_sha256="m", event_embedding_sha256="e", git_commit="g",
    )


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)
    return str(path)


def test_load_predictions_filters_tier_and_positive(tmp_path):
    path = _write_csv(tmp_path / "pred.csv", [
        _row("AAA1", "BBB1", "PT-1", 0, 0.9, 0.1),
        _row("AAA1", "BBB1", "PT-1", 1, 0.9, 0.1),
        _row("ZZZ9", "ZZY9", "PT-0", 0, 0.999, 0.001, y_true=0),   # negative: drop
        _row("CCC2", "DDD2", "PT-2", 0, 0.9, 0.1, tier="5shot"),   # other tier: drop
    ])
    df = load_predictions(path)
    assert list(df["drug_a"]) == ["AAA1", "AAA1"]
    assert df["y_true"].eq(1).all()


def test_aggregate_known_means_and_r(tmp_path):
    """Known seed-varying probs/uncertainties -> exact group means and r = p(1-u)."""
    probs1 = [0.80, 0.90, 0.70, 0.85, 0.95]
    us1 = [0.20, 0.15, 0.25, 0.10, 0.05]
    rows = [_row("AAA1", "BBB1", "PT-1", s, p, u) for s, (p, u) in enumerate(zip(probs1, us1))]
    rows += [_row("CCC2", "DDD2", "PT-2", s, 0.5, 0.3) for s in range(5)]
    agg = aggregate_candidates(load_predictions(_write_csv(tmp_path / "pred.csv", rows)))
    assert len(agg) == 2
    by_event = {r["event"]: r for _, r in agg.iterrows()}
    r1 = by_event["PT-1"]
    assert r1["prob_mean"] == pytest.approx(0.84)          # (0.8+0.9+0.7+0.85+0.95)/5
    assert r1["u_mean"] == pytest.approx(0.15)             # (0.2+0.15+0.25+0.1+0.05)/5
    assert r1["r"] == pytest.approx(0.84 * (1 - 0.15))     # 0.714
    r2 = by_event["PT-2"]
    assert r2["prob_mean"] == pytest.approx(0.5)
    assert r2["r"] == pytest.approx(0.5 * 0.7)


def test_rank_top10_cutoff_and_overlap_exclusion(tmp_path):
    """13 eligible pairs + 1 overlapping pair with the highest r: overlap dropped,
    exactly top-10 returned, ranks 1..10, r strictly descending."""
    rows = []
    for i in range(13):
        p = 0.50 + i * 0.03          # r = p * 0.9 rises with i
        rows += [_row(f"A{i:02d}", f"B{i:02d}", "PT-1", s, p, 0.1) for s in range(5)]
    # overlapping pair: highest r = 0.99*1.0 = 0.99, must be excluded
    rows += [_row("ZOVERLAP", "ZEXCLUDE", "PT-2", s, 0.99, 0.0) for s in range(5)]
    agg = aggregate_candidates(load_predictions(_write_csv(tmp_path / "pred.csv", rows)))
    overlap = {("ZEXCLUDE", "ZOVERLAP")}  # contract: sorted (canonical) IK14 tuple
    out = rank_candidates(agg, overlap, top_n=10)
    assert len(out) == 10
    assert list(out["rank"]) == list(range(1, 11))
    assert out["r"].is_monotonic_decreasing
    assert not (out["drug_a"].isin(["ZOVERLAP"]) | out["drug_b"].isin(["ZEXCLUDE"])).any()
    assert out.iloc[0]["drug_a"] == "A12"     # largest eligible r = (0.5+12*0.03)*0.9


def test_join_faers_order_independent(tmp_path):
    cand = pd.DataFrame([
        dict(rank=1, drug_a="AAA1", drug_b="BBB1", event="PT-1",
             prob_mean=0.9, u_mean=0.1, r=0.81),
        dict(rank=2, drug_a="CCC2", drug_b="DDD2", event="PT-2",   # reversed in ddi file
             prob_mean=0.8, u_mean=0.2, r=0.64),
    ])
    ddi = pd.DataFrame([
        dict(drug_a_ik14="AAA1", drug_b_ik14="BBB1", a_name="Drug A", b_name="Drug B",
             n_faers_reports=123.0, faers_prr_max_strict=4.5, faers_ror95_lcl_max_strict=3.2),
        dict(drug_a_ik14="DDD2", drug_b_ik14="CCC2", a_name="Drug D", b_name="Drug C",
             n_faers_reports=7.0, faers_prr_max_strict=9.9, faers_ror95_lcl_max_strict=8.8),
    ])
    out = join_faers(cand, ddi)
    assert list(out.columns) == ["rank", "drug_a", "drug_b", "a_name", "b_name", "event",
                                 "prob_mean", "u_mean", "r", "faers_prr_max_strict",
                                 "faers_ror95_lcl_max_strict", "n_faers_reports"]
    assert out.iloc[0]["a_name"] == "Drug A"
    assert out.iloc[0]["n_faers_reports"] == 123
    assert out.iloc[1]["a_name"] == "Drug D" and out.iloc[1]["b_name"] == "Drug C"
    assert out.iloc[1]["n_faers_reports"] == 7
    assert out.iloc[1]["faers_prr_max_strict"] == pytest.approx(9.9)


def test_build_esearch_term():
    assert build_esearch_term("Warfarin", "Aspirin") == (
        '"Warfarin"[All Fields] AND "Aspirin"[All Fields] AND (interaction OR adverse)')


def test_corroborated_heuristic():
    assert corroborated("Warfarin and aspirin interaction: a case report", "Warfarin", "Aspirin")
    assert corroborated("Adverse reaction to combined warfarin aspirin therapy", "Warfarin", "Aspirin")
    assert not corroborated("Warfarin monotherapy: a review", "Warfarin", "Aspirin")
    assert not corroborated("Aspirin and clopidogrel combination therapy", "Warfarin", "Aspirin")
    assert not corroborated("", "Warfarin", "Aspirin")


def test_gather_evidence_stubbed_fetch():
    calls = []

    def fake_fetch(a_name, b_name):
        calls.append((a_name, b_name))
        if a_name == "Warfarin":
            return [("11111111", f"{a_name} and {b_name} interaction: a case report")]
        return [("22222222", f"{a_name} alone: a review")]

    cands = pd.DataFrame([
        dict(rank=1, drug_a="A1", drug_b="B1", a_name="Warfarin", b_name="Aspirin",
             event="PT-1", prob_mean=0.9, u_mean=0.1, r=0.81,
             faers_prr_max_strict=4.5, faers_ror95_lcl_max_strict=3.2, n_faers_reports=123),
        dict(rank=2, drug_a="A2", drug_b="B2", a_name="Metformin", b_name="Cimetidine",
             event="PT-2", prob_mean=0.8, u_mean=0.2, r=0.64,
             faers_prr_max_strict=1.5, faers_ror95_lcl_max_strict=1.2, n_faers_reports=9),
    ])
    ev = gather_evidence(cands, fetch=fake_fetch)
    assert len(ev) == 2
    assert ev[0]["corroborated"] is True
    assert ev[0]["pmids"] == ["11111111"]
    assert ev[1]["corroborated"] is False
    assert calls == [("Warfarin", "Aspirin"), ("Metformin", "Cimetidine")]


def test_gather_evidence_empty_hits_not_corroborated():
    def no_hits(a_name, b_name):
        return []
    cands = pd.DataFrame([
        dict(rank=1, drug_a="A1", drug_b="B1", a_name="Warfarin", b_name="Aspirin",
             event="PT-1", prob_mean=0.9, u_mean=0.1, r=0.81,
             faers_prr_max_strict=None, faers_ror95_lcl_max_strict=None, n_faers_reports=None),
    ])
    ev = gather_evidence(cands, fetch=no_hits)
    assert ev[0]["pmids"] == []
    assert ev[0]["corroborated"] is False


def test_render_md_header_has_count_and_sections(tmp_path):
    cands = pd.DataFrame([
        dict(rank=1, drug_a="A1", drug_b="B1", a_name="Warfarin", b_name="Aspirin",
             event="PT-1", prob_mean=0.9, u_mean=0.1, r=0.81,
             faers_prr_max_strict=4.5, faers_ror95_lcl_max_strict=3.2, n_faers_reports=123),
        dict(rank=2, drug_a="A2", drug_b="B2", a_name="Metformin", b_name="Cimetidine",
             event="PT-2", prob_mean=0.8, u_mean=0.2, r=0.64,
             faers_prr_max_strict=1.5, faers_ror95_lcl_max_strict=1.2, n_faers_reports=9),
    ])
    evidence = [
        dict(rank=1, pmids=["11111111"], titles=["Warfarin and aspirin interaction: a case report"],
             corroborated=True),
        dict(rank=2, pmids=[], titles=[], corroborated=False),
    ]
    md = render_evidence_md(cands, evidence, src_csv="predictions_x.csv", n_overlap=304)
    assert "1/2" in md
    assert "## Rank 1" in md and "## Rank 2" in md
    assert "11111111" in md and "interaction" in md
    assert "Warfarin" in md and "Aspirin" in md
    assert "304" in md


def test_pubmed_fetch_endpoint_wiring(monkeypatch):
    """esearch then esummary must hit the right endpoints (regression: both calls
    used to go to esearch.fcgi, which silently returns empty titles). No network."""
    import time
    import urllib.request
    from case_study_ext import pubmed_fetch

    captured = []

    class FakeResp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    esearch_body = json.dumps({"esearchresult": {"count": "1", "idlist": ["36034061"]}}).encode()
    esummary_body = json.dumps({"result": {"uids": ["36034061"],
                                           "36034061": {"title": "Fixed Drug Eruption."}}}).encode()

    def fake_urlopen(req, timeout=60):
        captured.append(req.full_url)
        if "esearch.fcgi" in req.full_url:
            return FakeResp(esearch_body)
        return FakeResp(esummary_body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = pubmed_fetch("Hydrocortisone acetate", "Celecoxib", sleep=0.4)
    assert out == [("36034061", "Fixed Drug Eruption.")]
    assert "esearch.fcgi" in captured[0]
    assert "esummary.fcgi" in captured[1]
    assert "id=36034061" in captured[1]
    assert sleeps == [0.4, 0.4]


def test_parse_esearch_ids():
    payload = json.dumps({"esearchresult": {"count": "2", "idlist": ["39853509", "39351090"]}})
    assert parse_esearch_ids(payload) == ["39853509", "39351090"]


def test_parse_esummary_titles():
    payload = json.dumps({"result": {
        "39853509": {"title": "A title."},
        "39351090": {"title": "B title."},
        "uids": ["39853509", "39351090"],
    }})
    assert parse_esummary_titles(payload) == {"39853509": "A title.", "39351090": "B title."}
