#!/usr/bin/env bash
# train_all_seeds.sh — five-seed training orchestrator (PharDDIE 1/5-shot + EviDDIE 0-shot)
# Saves every checkpoint to the exact paths required by the exporters:
#   PharDDIE:  models/dataset1/models_drugbank_{1,5}shot_str_seed{seed}/bestmodel
#   EviDDIE:   models/dataset1/eviddie_0shot_seed{seed}/bestmodel{,_G}
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-19940419 20230801 20240115 20240520 20240910}"
SHOTS="${SHOTS:-1 5}"
MAX_BATCHES="${MAX_BATCHES:-40000}"
SKIP_PHARDDIE="${SKIP_PHARDDIE:-0}"
SKIP_EVIDDIE="${SKIP_EVIDDIE:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_cmd() {
    local name="$1"; local workdir="$2"; shift 2
    echo "=== $name ==="
    (cd "$REPO_ROOT/$workdir" && "$@") || {
        echo "ABORTING: $name failed" >&2
        exit 1
    }
    echo "PASS: $name"
}

# ---- PharDDIE 1/5-shot, five seeds ----
if [ "$SKIP_PHARDDIE" != "1" ]; then
    for shot in $SHOTS; do
        for seed in $SEEDS; do
            ckpt="models/dataset1/models_drugbank_${shot}shot_str_seed${seed}/bestmodel"
            if [ -f "$ckpt" ]; then
                echo "[SKIP] $ckpt already exists"
                continue
            fi
            run_cmd "PharDDIE ${shot}-shot seed ${seed}" "PharDDIE" \
                python pharddie_trainer.py --dataset dataset1 \
                --prefix "dataset1/models_drugbank_${shot}shot_str_seed${seed}" \
                --seed "$seed" --few "$shot" --train_few "$shot" \
                --max_batches "$MAX_BATCHES"
        done
    done
fi

# ---- EviDDIE 0-shot, five seeds ----
if [ "$SKIP_EVIDDIE" != "1" ]; then
    for seed in $SEEDS; do
        ckpt="models/dataset1/eviddie_0shot_seed${seed}/bestmodel"
        if [ -f "$ckpt" ]; then
            echo "[SKIP] $ckpt already exists"
            continue
        fi
        run_cmd "EviDDIE 0-shot seed ${seed}" "EviDDIE" \
            python eviddie_trainer.py --dataset dataset1 \
            --prefix dataset1/eviddie_0shot --seed "$seed" \
            --max_batches 20000
    done
fi

echo "=== TRAINING COMPLETE ==="
