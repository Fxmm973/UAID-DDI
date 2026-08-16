#!/usr/bin/env python
# coding=utf-8
"""
run_eviddie_ablation.py - One-command EviDDIE frozen-backbone ablation.

  1) verify the backbone checkpoints (pharddie_best.pt / bestmodels_G)
  2) train the four head variants (eviddie_train_ablation.py; frozen backbone,
     5000 iters per variant)
  3) evaluate the three exported variants under the unified protocol
     (eval_eviddie_ablation.py; fixed manifest seed 19940419)

Usage (from the EviDDIE directory): python run_eviddie_ablation.py
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = [
    'models/dataset1/pharddie_best.pt',
    'models/dataset1/bestmodels_G',
]
HEADS = [
    'models/dataset1/fc_softmax.pt',
    'models/dataset1/fc_evi_no_evi.pt',
    'models/dataset1/fc_full_evi.pt',
]


def step(cmd, name):
    print(f'\n===== {name} =====\n  {" ".join(cmd)}')
    r = subprocess.run(cmd, cwd=BASE)
    if r.returncode != 0:
        raise SystemExit(f'{name} failed (exit {r.returncode}); see the log above')


def main():
    missing = [p for p in REQUIRED if not os.path.exists(os.path.join(BASE, p))]
    if missing:
        raise SystemExit(f'Missing checkpoints: {missing}\n'
                         f'pharddie_best.pt = a PharDDIE 1-shot bestmodel; bestmodels_G = the BSA generator.')

    step([sys.executable, 'eviddie_train_ablation.py'], 'Step 1/2: train four head variants (frozen backbone)')
    miss_heads = [p for p in HEADS if not os.path.exists(os.path.join(BASE, p))]
    if miss_heads:
        raise SystemExit(f'Missing head files: {miss_heads}; check the Step 1 log')

    step([sys.executable, 'eval_eviddie_ablation.py'], 'Step 2/2: unified-protocol evaluation of the three variants')
    print('\nDone. Results: results/eviddie_ablation_results.csv + results/ablation_curves.csv')


if __name__ == '__main__':
    main()
