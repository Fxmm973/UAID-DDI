# external/ext_summary_table.py
# Task 9: external-validation summary table.
# Groups per-tier predictions by train_seed, computes pooled AUROC/AUPRC/ACC/F1
# (threshold 0.5) and event-macro F1 (mean over per-event F1; events with only
# one class in y_true are excluded with a count), then reports mean +/- SD of
# each metric across train seeds (sample SD, ddof=1). A no-skill reference row
# (constant p=0.5 -> AUROC=0.5, AUPRC=positive prevalence, ACC=0.5 on balanced
# sets, F1 of the constant-positive predictor with zero_division=0) is emitted
# per tier x variant.
#
# Usage:  python external/ext_summary_table.py
# Outputs: external/outputs/ext_validation_table.csv + .txt
"""External-validation summary table (mean±SD across train seeds)."""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CSV_GLOB = os.path.join(OUT_DIR, "predictions_rxpairevid_eviddie_*_0shot*.csv")

METRIC_COLS = ["auroc", "auprc", "acc", "f1", "macro_f1"]
MEAN_SD_COLS = [f"{m}_{agg}" for m in METRIC_COLS for agg in ("mean", "sd")]


def infer_variant(path):
    """Native tier if the filename ends with '_native', else tail-corrupted."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    return "native" if stem.endswith("_native") else "tail-corrupted"


def _roc_auc_guarded(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan  # single class: undefined; excluded downstream
    return roc_auc_score(y_true, prob)


def _avg_prec_guarded(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, prob)


def pooled_metrics(df, noskill=False):
    """Pooled metrics over one (train_seed) subset. noskill=True predicts
    constant p=0.5 (y_pred=1 at threshold 0.5) as a no-skill reference."""
    y = df["y_true"].to_numpy().astype(int)
    if noskill:
        p = np.full(len(df), 0.5)
        yp = np.ones(len(df), dtype=int)
    else:
        p = df["prob"].to_numpy().astype(float)
        yp = df["y_pred"].to_numpy().astype(int)

    ev_f1, n_excluded = [], 0
    for _, g in df.groupby("event_type", sort=False):
        gy = g["y_true"].to_numpy().astype(int)
        if len(np.unique(gy)) < 2:
            n_excluded += 1  # single-class event: F1 = NaN, excluded from macro mean
            continue
        gyp = np.ones(len(g), dtype=int) if noskill else g["y_pred"].to_numpy().astype(int)
        ev_f1.append(f1_score(gy, gyp, zero_division=0))

    return {
        "auroc": _roc_auc_guarded(y, p),
        "auprc": _avg_prec_guarded(y, p),
        "acc": accuracy_score(y, yp),
        "f1": f1_score(y, yp, zero_division=0),
        "macro_f1": float(np.mean(ev_f1)) if ev_f1 else np.nan,
        "n_events": len(ev_f1),
        "n_excluded": n_excluded,
    }


def _mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return (float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)))


def _group_row(df, variant, noskill=False):
    """Aggregate pooled metrics across the train seeds of one file."""
    per_seed = [pooled_metrics(g, noskill=noskill)
                for _, g in df.groupby("train_seed", sort=False)]
    row = {
        "tier": str(df["tier"].iloc[0]),
        "variant": variant,
        "method": str(df["method"].iloc[0]),
        "shot": int(df["shot"].iloc[0]),
        "is_noskill": int(noskill),
        "n_seeds": len(per_seed),
        "n_samples": int(len(df)),
    }
    for m in METRIC_COLS:
        row[f"{m}_mean"], row[f"{m}_sd"] = _mean_sd([r[m] for r in per_seed])
    row["macro_f1_n_events"] = float(np.mean([r["n_events"] for r in per_seed]))
    row["macro_f1_n_excluded"] = float(np.mean([r["n_excluded"] for r in per_seed]))
    return row


def build_table(csv_paths):
    """One row per (tier x variant) file plus its no-skill reference row."""
    rows = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df["tier"].nunique() != 1:
            raise ValueError(f"mixed tiers in {path}: {df['tier'].unique()}")
        variant = infer_variant(path)
        rows.append(_group_row(df, variant, noskill=False))
        rows.append(_group_row(df, variant, noskill=True))
    table = pd.DataFrame(rows)
    return table.sort_values(
        ["tier", "variant", "is_noskill"],
        key=lambda s: s.astype(str) if s.name == "variant" else s,
    ).reset_index(drop=True)


def _fmt(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}"


def format_txt(table):
    """Paper-table layout (pipe-separated, LaTeX-pasteable)."""
    lines = ["Setting | Variant | AUROC | AUPRC | ACC | F1 | macro-F1 | n_events",
             "--------|---------|-------|-------|-----|----|----------|---------"]
    for _, r in table.iterrows():
        variant = f"{r['variant']} (no-skill)" if r["is_noskill"] else r["variant"]
        cells = [str(r["tier"]), variant]
        for m in METRIC_COLS:
            cells.append(_fmt(r[f"{m}_mean"], r[f"{m}_sd"]) if not np.isnan(r[f"{m}_mean"]) else "n/a")
        cells.append(f"{r['macro_f1_n_events']:g}")
        lines.append(" | ".join(cells))
    lines.append("")
    lines.append("# n_events: number of events entering the macro-F1 mean (events with a single")
    lines.append("# class in y_true are excluded); no-skill = constant p=0.5 predictor (AUROC=0.5,")
    lines.append("# AUPRC=positive prevalence, ACC=0.5 on balanced sets). Metrics: mean ± SD across")
    lines.append("# train seeds (ddof=1).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=sorted(glob.glob(CSV_GLOB)),
                        help="prediction CSVs (default: external/outputs/predictions_rxpairevid_eviddie_*_0shot*.csv)")
    parser.add_argument("--out-csv", default=os.path.join(OUT_DIR, "ext_validation_table.csv"))
    parser.add_argument("--out-txt", default=os.path.join(OUT_DIR, "ext_validation_table.txt"))
    args = parser.parse_args()

    table = build_table(args.inputs)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    txt = format_txt(table)
    with open(args.out_txt, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"\nWrote {args.out_csv} and {args.out_txt}")


if __name__ == "__main__":
    main()
