# external/tests/test_ext_summary_table.py
# Task 9: external validation summary table — hand-computable metric assertions.
import math
import numpy as np
import pandas as pd
import pytest

from ext_summary_table import (
    build_table,
    format_txt,
    infer_variant,
    pooled_metrics,
)

SEED_A = 19940419   # perfect ranking -> all metrics 1.0
SEED_B = 20230801   # perfectly reversed ranking -> all metrics 0.0
SD1 = math.sqrt(0.5)  # sample SD (ddof=1) of [1.0, 0.0]

# 2 seeds x 2 events x 4 samples (16 rows, 8 per seed).
SYN_ROWS = [
    # seed A: probs 0.1,0.4,0.6,0.9 align with labels 0,0,1,1 (perfect AUC)
    (SEED_A, "E1", 0, 0.1), (SEED_A, "E1", 0, 0.4),
    (SEED_A, "E1", 1, 0.6), (SEED_A, "E1", 1, 0.9),
    (SEED_A, "E2", 0, 0.1), (SEED_A, "E2", 0, 0.4),
    (SEED_A, "E2", 1, 0.6), (SEED_A, "E2", 1, 0.9),
    # seed B: reversed ranking (perfectly anti-correlated)
    (SEED_B, "E1", 0, 0.9), (SEED_B, "E1", 0, 0.6),
    (SEED_B, "E1", 1, 0.4), (SEED_B, "E1", 1, 0.1),
    (SEED_B, "E2", 0, 0.9), (SEED_B, "E2", 0, 0.6),
    (SEED_B, "E2", 1, 0.4), (SEED_B, "E2", 1, 0.1),
]


def make_synth_csv(path, rows=SYN_ROWS, tier="1shot", extra_event=None):
    """Write a synthetic predictions CSV (plus optional single-class event)."""
    recs = []
    for seed, ev, y, p in rows:
        recs.append({
            "train_seed": seed, "eval_seed": 19940419, "tier": tier,
            "shot": 0, "method": "EviDDIE", "event_type": ev,
            "drug_a": "D1", "drug_b": "D2", "y_true": y,
            "y_pred": int(p >= 0.5), "prob": p, "uncertainty": 0.0,
        })
    if extra_event:  # (seed, event, [y_true...], [prob...])
        seed, ev, ys, ps = extra_event
        for y, p in zip(ys, ps):
            recs.append({
                "train_seed": seed, "eval_seed": 19940419, "tier": tier,
                "shot": 0, "method": "EviDDIE", "event_type": ev,
                "drug_a": "D1", "drug_b": "D2", "y_true": y,
                "y_pred": int(p >= 0.5), "prob": p, "uncertainty": 0.0,
            })
    pd.DataFrame(recs).to_csv(path, index=False)
    return path


def test_variant_inference_from_filename():
    assert infer_variant("predictions_rxpairevid_eviddie_1shot_0shot.csv") == "tail-corrupted"
    assert infer_variant("predictions_rxpairevid_eviddie_1shot_0shot_native.csv") == "native"
    assert infer_variant("external/outputs/x_5shot_0shot_native.csv") == "native"


def test_pooled_metrics_match_hand_calculation():
    df = pd.DataFrame(SYN_ROWS, columns=["train_seed", "event_type", "y_true", "prob"])
    df["y_pred"] = (df["prob"] >= 0.5).astype(int)

    m_a = pooled_metrics(df[df.train_seed == SEED_A])
    assert m_a["auroc"] == pytest.approx(1.0)
    assert m_a["auprc"] == pytest.approx(1.0)
    assert m_a["acc"] == pytest.approx(1.0)
    assert m_a["f1"] == pytest.approx(1.0)
    assert m_a["macro_f1"] == pytest.approx(1.0)
    assert m_a["n_events"] == 2 and m_a["n_excluded"] == 0

    m_b = pooled_metrics(df[df.train_seed == SEED_B])
    assert m_b["auroc"] == pytest.approx(0.0)
    # AP of a perfectly reversed ranking on 4 pos / 4 neg with distinct scores:
    # AP = sum(Delta_recall * precision) = (0.5-0)*(2/6) + (1-0.5)*(4/8) = 5/12
    assert m_b["auprc"] == pytest.approx(5.0 / 12.0)
    assert m_b["acc"] == pytest.approx(0.0)
    assert m_b["f1"] == pytest.approx(0.0)
    assert m_b["macro_f1"] == pytest.approx(0.0)


def test_grouped_table_matches_hand_calculation(tmp_path):
    path = make_synth_csv(str(tmp_path / "predictions_rxpairevid_eviddie_1shot_0shot.csv"))
    table = build_table([path])
    row = table.iloc[0]

    assert row["tier"] == "1shot"
    assert row["variant"] == "tail-corrupted"
    assert row["n_seeds"] == 2 and row["n_samples"] == 16
    for m in ["auroc", "acc", "f1", "macro_f1"]:
        assert row[f"{m}_mean"] == pytest.approx(0.5)
        assert row[f"{m}_sd"] == pytest.approx(SD1)
    # auprc: per-seed values [1.0, 5/12] -> mean 17/24, sd (7/24)*sqrt(2)
    assert row["auprc_mean"] == pytest.approx(17.0 / 24.0)
    assert row["auprc_sd"] == pytest.approx(7.0 / 24.0 * math.sqrt(2))
    assert row["macro_f1_n_events"] == pytest.approx(2.0)
    assert row["macro_f1_n_excluded"] == pytest.approx(0.0)


def test_single_class_event_excluded_with_count(tmp_path):
    path = make_synth_csv(
        str(tmp_path / "predictions_rxpairevid_eviddie_5shot_0shot.csv"), tier="5shot",
        extra_event=(SEED_A, "E3", [1, 1, 1, 1], [0.2, 0.3, 0.7, 0.8]),
    )
    table = build_table([path])
    row = table.iloc[0]
    # seed A: E1+E2 two-class, E3 single-class excluded (n_events=2, n_excluded=1);
    # seed B: E1+E2 two-class (n_events=2, n_excluded=0)
    assert row["macro_f1_n_events"] == pytest.approx(2.0)
    assert row["macro_f1_n_excluded"] == pytest.approx(0.5)
    # macro-F1 still 0.5 mean / SD1 across seeds (E1 perfect vs anti-perfect)
    assert row["macro_f1_mean"] == pytest.approx(0.5)
    assert row["macro_f1_sd"] == pytest.approx(SD1)


def test_noskill_reference_row(tmp_path):
    path = make_synth_csv(str(tmp_path / "predictions_rxpairevid_eviddie_1shot_0shot.csv"))
    table = build_table([path])
    ref = table[table["is_noskill"] == 1]
    assert len(ref) == 1
    r = ref.iloc[0]
    # constant p=0.5 on the balanced pooled set
    assert r["acc_mean"] == pytest.approx(0.5)
    assert r["auroc_mean"] == pytest.approx(0.5)
    assert r["auprc_mean"] == pytest.approx(0.5)   # = positive prevalence
    assert r["f1_mean"] == pytest.approx(2.0 / 3.0)  # constant-positive F1
    assert r["auroc_sd"] == pytest.approx(0.0)      # deterministic across seeds
    # no-skill macro-F1 must come from constant-positive per-event predictions:
    # each event y=[0,0,1,1] -> F1 = 2*2/(2+4) = 2/3, so macro = 2/3 for every seed
    assert r["macro_f1_mean"] == pytest.approx(2.0 / 3.0)
    assert r["macro_f1_sd"] == pytest.approx(0.0)


def test_format_txt_contains_rows_for_latex(tmp_path):
    path = make_synth_csv(str(tmp_path / "predictions_rxpairevid_eviddie_1shot_0shot.csv"))
    table = build_table([path])
    txt = format_txt(table)
    assert "Setting" in txt and "Variant" in txt and "AUROC" in txt
    assert "tail-corrupted" in txt
    assert "no-skill" in txt
    assert "0.500" in txt and "0.707" in txt
