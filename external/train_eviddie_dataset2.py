#!/usr/bin/env python
# coding=utf-8
"""
Task 13: EviDDIE Dataset-2 (Lin protocol) training wrapper.

Trains EviDDIE on EviDDIE/dataset2 with dataset2-built molecular graphs and the
exact hyperparameters of the README formal training (eviddie_trainer.py), while
NEVER modifying any existing repo file. One (seed, prefix) run per invocation.

WHY A WRAPPER (the dataset1-hardcode problem)
--------------------------------------------
The stock training path builds every drug-pair batch through
eviddie_dataloader.DrugDataset, whose __init__/collate_fn read module globals
    MOL_EDGE_LIST_FEAT_MTX  (drug_id -> (edge_index, node_feats, edge_feats))
    id_fp / id_desc         (per-drug fingerprint / descriptor tensors)
bound at import time by `from shared.preprocess import *`.
shared/preprocess.py HARDCODES PharDDIE/dataset1/drug_smiles.csv at import
time, so running the stock trainer on dataset2 would silently use dataset1's
molecule graphs (dataset2-only drugs would be filtered out of every batch).
(Trainer.all_drug_data / drug_num_node_indices are vestigial: initialized to {}
and never populated; train_generate ignores them.)

WHAT THIS WRAPPER DOES
----------------------
1. Path resolution is repo-root-relative (R7): the repo root is located by
   walking up from this file (no dependence on the launch directory).
2. Rebuilds the molecule registry from EviDDIE/dataset2/drug_smiles.csv using
   the same rdkit pipeline as the T8 clone external/eviddie_export_ext.py
   (get_mol_edge_list_and_feat_mtx for atom/bond features + chemprop-style
   morgan 2048-bit fingerprint, validated bit-exact there against
   PharDDIE/fp/features/morgan_dataset1.npz). id_desc = zeros(200), as in the
   clone and in shared/preprocess.py.
3. Rebinds shared.preprocess.* data globals to the dataset2 structures BEFORE
   eviddie_dataloader is imported, and re-asserts the rebinding on the
   eviddie_dataloader namespace afterwards (star-import copies are by value),
   so DrugDataset resolves every dataset2 drug id.
4. P0-4 dev checkpoint selection needs neg_manifests/dev_seed19940419_negatives.json,
   which the dataset2 copy does not ship (only test2_seed* files). dataset2 is
   fully git-tracked, so the wrapper generates the dev manifest ONCE, into
   external/outputs/train_ds2_dev_manifest/, with the same tail-corruption
   algorithm as external/neg_manifest_ext.py (deterministic; random.seed(seed)
   inside the generator), and patches eviddie_trainer.load_fixed_event_rows
   with a fallback that resolves a missing dev manifest to that location.
5. Redirects the trainer's side outputs out of EviDDIE/: tensorboard SummaryWriter
   ('logs/' + prefix) and ExperimentRecorder result files are redirected into
   external/outputs/train_logs_ds2/. Checkpoints stay at the stock location
   models/{prefix}_seed{seed}[bestmodel{, _G, _D}] relative to EviDDIE/ (the
   wrapper chdirs to EviDDIE/, exactly like the stock entry). Logging goes to
   external/outputs/train_logs_ds2/log-{prefix}_seed{seed}.txt + stdout.
6. Injects the two module-level globals eviddie_trainer normally binds in its
   __main__ block: `loss_fn` (EvidentialLoss, annealing 10000) and `args`
   (referenced by train_standard at bestmodel-save time via getattr(args, ...)).

USAGE (from anywhere)
---------------------
    <conda env PharDDIE python> external/train_eviddie_dataset2.py --seed 19940419 --max-batches 20000
    # smoke: --max-batches 300 --prefix eviddie_ds2_smoke
"""
import argparse
import hashlib
import json
import logging
import os
import random
import sys

DEV_MANIFEST_SEED = 19940419  # stock --eval-manifest-seed default (P0-4 fixed manifest)


def find_repo_root(start_dir):
    """R7: walk up from this file until the dir containing EviDDIE/, shared/, external/."""
    d = os.path.abspath(start_dir)
    while True:
        if all(os.path.isdir(os.path.join(d, x)) for x in ('EviDDIE', 'shared', 'external')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError('Could not locate repo root (EviDDIE/shared/external) above %s' % start_dir)
        d = parent


def main():
    ap = argparse.ArgumentParser(
        description='Train EviDDIE on EviDDIE/dataset2 with dataset2 molecular graphs (one seed per run)')
    ap.add_argument('--seed', type=int, default=19940419, help='training seed (default 19940419)')
    ap.add_argument('--max-batches', type=int, default=20000,
                    help='training iterations (default 20000; use e.g. 300 for a smoke run)')
    ap.add_argument('--device-id', type=int, default=None,
                    help='CUDA device index to restrict to (CUDA_VISIBLE_DEVICES, set before torch import)')
    ap.add_argument('--prefix', default='eviddie_ds2',
                    help='checkpoint/log prefix; models land in models/{prefix}_seed{seed} (default eviddie_ds2)')
    cli = ap.parse_args()

    _HERE = os.path.dirname(os.path.abspath(__file__))
    REPO = find_repo_root(_HERE)
    EVIDDIE = os.path.join(REPO, 'EviDDIE')
    DATASET = os.path.join(EVIDDIE, 'dataset2')
    LOG_DIR = os.path.join(REPO, 'external', 'outputs', 'train_logs_ds2')
    MANIFEST_DIR = os.path.join(REPO, 'external', 'outputs', 'train_ds2_dev_manifest')
    for _p in (LOG_DIR, MANIFEST_DIR):
        os.makedirs(_p, exist_ok=True)
    for _p in (REPO, EVIDDIE, os.path.join(REPO, 'external')):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    if cli.device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.device_id)

    import numpy as np
    import torch
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # ---- logging (stock trainer format, but into external/outputs) ----
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    _fmt = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    _fh = logging.FileHandler(os.path.join(LOG_DIR, f'log-{cli.prefix}_seed{cli.seed}.txt'),
                              mode='w', encoding='utf-8')
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)

    logging.info('[DS2-TRAIN] repo root: %s', REPO)
    logging.info('[DS2-TRAIN] dataset: %s | seed: %d | max_batches: %d | prefix: %s | device_id: %s',
                 DATASET, cli.seed, cli.max_batches, cli.prefix, cli.device_id)
    logging.info('[DS2-TRAIN] logs: %s | dev-manifest dir: %s', LOG_DIR, MANIFEST_DIR)

    # =====================================================================
    # 1. dataset2 molecular-graph registry (mirror of external/eviddie_export_ext.py)
    # =====================================================================
    import shared.preprocess as preprocess  # pure functions; data globals rebound below
    df_drugs = pd.read_csv(os.path.join(DATASET, 'drug_smiles.csv'),
                           dtype={'drug_id': 'string', 'smiles': 'string'})
    n_dup = int(df_drugs['drug_id'].duplicated().sum())
    if n_dup:
        raise RuntimeError(f'dataset2 drug_smiles.csv has {n_dup} duplicate drug_ids')

    mol_graphs, feats, fps, descs = {}, {}, {}, {}
    n_parse_fail = 0
    for _, row in df_drugs.iterrows():
        d, smi = str(row['drug_id']), str(row['smiles'])
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            n_parse_fail += 1
            logging.warning('[DS2-GRAPHS] unparseable SMILES for %s; triples with it will be dropped', d)
            continue
        f = preprocess.get_mol_edge_list_and_feat_mtx(mol)
        if f is None:
            n_parse_fail += 1
            continue
        mol_graphs[d] = mol
        feats[d] = f
        fps[d] = torch.from_numpy(np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False),
            dtype=np.float32))
        descs[d] = torch.zeros(200, dtype=torch.float32)

    # rebind the data globals the training path reads (must precede eviddie_dataloader import)
    preprocess.MOL_EDGE_LIST_FEAT_MTX = feats
    preprocess.id_fp = fps
    preprocess.id_desc = descs
    preprocess.drug_to_mol_graph = mol_graphs
    preprocess.DRUG_TO_INDX_DICT = {d: i for i, d in enumerate(df_drugs['drug_id'])}
    preprocess.DRUG_INDX_NAME_DICT = {i: d for i, d in enumerate(df_drugs['drug_id'])}

    logging.info('[DS2-GRAPHS] dataset2 drug registry: %d drugs in CSV, %d molecule graphs built '
                 '(parse-fail %d, dup %d)', len(df_drugs), len(feats), n_parse_fail, n_dup)
    # assertion: dataset2-only drug ids are present in the graph dict (the stock
    # path via dataset1 would silently lack them)
    ds1 = pd.read_csv(os.path.join(REPO, 'PharDDIE', 'dataset1', 'drug_smiles.csv'),
                      dtype={'drug_id': 'string'})
    ds2_only = sorted(set(df_drugs['drug_id']) - set(ds1['drug_id']))
    if not ds2_only:
        raise RuntimeError('dataset2 has no drug id outside dataset1; cannot prove graph substitution')
    missing = [d for d in ds2_only if d not in feats]
    if missing:
        raise RuntimeError(f'{len(missing)} dataset2-only drugs missing from the molecule registry: {missing[:5]}')
    logging.info('[DS2-GRAPHS] %d dataset2-only drugs (not in dataset1) all present in graph dict, '
                 'e.g. %s (graph edge_index shape %s)', len(ds2_only), ds2_only[0],
                 tuple(feats[ds2_only[0]][0].shape))

    # =====================================================================
    # 2. P0-4 dev manifest (missing in dataset2; generated once under external/)
    # =====================================================================
    from neg_manifest_ext import generate_manifest_ext
    dev_manifest_path = os.path.join(MANIFEST_DIR, f'dev_seed{DEV_MANIFEST_SEED}_negatives.json')
    if not os.path.exists(dev_manifest_path):
        dev_tasks = json.load(open(os.path.join(DATASET, 'dev_tasks.json'), encoding='utf-8'))
        rel2cand = json.load(open(os.path.join(DATASET, 'rel2candidates.json'), encoding='utf-8'))
        dev_manifest = generate_manifest_ext(dev_tasks, rel2cand, seed=DEV_MANIFEST_SEED)
        n_pos = sum(len(v) for v in dev_tasks.values())
        n_neg = sum(len(v) for v in dev_manifest.values())
        if n_neg != n_pos:
            raise RuntimeError(f'dev manifest entry count {n_neg} != dev positives {n_pos}')
        with open(dev_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(dev_manifest, f, indent=2, ensure_ascii=False)
        sha = hashlib.sha256(open(dev_manifest_path, 'rb').read()).hexdigest()
        logging.warning('[DS2-MANIFEST] EviDDIE/dataset2/neg_manifests has no dev_seed* files; '
                        'generated deterministic dev manifest at %s (sha256=%s...)', dev_manifest_path, sha[:12])
    else:
        logging.info('[DS2-MANIFEST] dev manifest present: %s', dev_manifest_path)

    # =====================================================================
    # 3. import the stock training stack (after the rebinding), then re-assert
    # =====================================================================
    import eviddie_dataloader
    import eviddie_matcher      # noqa: F401  (import side effects)
    import eviddie_trainer
    # star imports copy by value; re-assert the ds2 structures on the consumer module
    eviddie_dataloader.MOL_EDGE_LIST_FEAT_MTX = feats
    eviddie_dataloader.id_fp = fps
    eviddie_dataloader.id_desc = descs

    # ---- P0-4 dev-manifest fallback (external/ dir; no writes inside EviDDIE/dataset2) ----
    _orig_load_fixed_event_rows = eviddie_trainer.load_fixed_event_rows

    def _load_fixed_event_rows_ds2(dataset_dir, split='dev', manifest_seed=DEV_MANIFEST_SEED):
        primary = os.path.join(dataset_dir, 'neg_manifests', f'{split}_seed{manifest_seed}_negatives.json')
        if os.path.exists(primary):
            return _orig_load_fixed_event_rows(dataset_dir, split=split, manifest_seed=manifest_seed)
        if split != 'dev':
            raise FileNotFoundError(primary)
        fallback = os.path.join(MANIFEST_DIR, f'{split}_seed{manifest_seed}_negatives.json')
        if not os.path.exists(fallback):
            raise FileNotFoundError(f'dev manifest missing: {primary} and fallback {fallback}')
        # same validation/row-building contract as shared/eval_manifest.load_fixed_event_rows
        tasks = json.load(open(os.path.join(dataset_dir, f'{split}_tasks.json'), encoding='utf-8'))
        manifest = json.load(open(fallback, encoding='utf-8'))
        rows = []
        for event, positives in tasks.items():
            entries = manifest.get(event)
            if entries is None:
                raise ValueError(f'Missing manifest event: {event}')
            neg_by_positive = {}
            for head, positive_tail, negative_tail, relation in entries:
                key = (head, relation, positive_tail)
                if key in neg_by_positive:
                    raise ValueError(f'Duplicate manifest key: {key}')
                neg_by_positive[key] = negative_tail
            if len(neg_by_positive) != len(positives):
                raise ValueError(f'{event}: {len(positives)} positives but {len(neg_by_positive)} manifest entries')
            for head, relation, tail in positives:
                key = (head, relation, tail)
                if key not in neg_by_positive:
                    raise ValueError(f'Manifest does not match positive: {key}')
                rows.append((event, head, relation, tail, 1))
                rows.append((event, head, relation, neg_by_positive[key], 0))
        sha = hashlib.sha256(open(fallback, 'rb').read()).hexdigest()
        logging.info('[DS2-MANIFEST] dev rows (%d positives) served from external fallback manifest '
                     '(sha256=%s...)', len(rows) // 2, sha[:12])
        return rows, sha

    eviddie_trainer.load_fixed_event_rows = _load_fixed_event_rows_ds2

    # ---- redirect side outputs into external/outputs/train_logs_ds2/ ----
    from tensorboardX import SummaryWriter
    from eviddie_recorder import ExperimentRecorder

    class _Ds2Writer(SummaryWriter):
        def __init__(self, logdir):
            name = os.path.basename(str(logdir).replace('\\', '/'))
            super().__init__(os.path.join(LOG_DIR, 'tensorboard', name))

    class _Ds2Recorder(ExperimentRecorder):
        def __init__(self, project_name='ZetaDDIE', result_file='result.txt'):
            super().__init__(project_name=project_name,
                             result_file=os.path.join(LOG_DIR, f'result_{cli.prefix}_seed{cli.seed}.txt'))

    eviddie_trainer.SummaryWriter = _Ds2Writer
    eviddie_trainer.ExperimentRecorder = _Ds2Recorder

    # =====================================================================
    # 4. trainer construction / training loop (mirror of eviddie_trainer.py __main__)
    # =====================================================================
    _old_argv = sys.argv
    sys.argv = ['train_eviddie_dataset2.py', '--dataset', DATASET, '--prefix', cli.prefix,
                '--seed', str(cli.seed), '--max_batches', str(cli.max_batches)]
    from eviddie_args import read_options
    args = read_options()
    sys.argv = _old_argv
    # all other hyperparameters stay at eviddie_args defaults = README formal
    # training (batch_size 256, few 10, train_few 10, lr 1e-3, eval_every 1000,
    # semantic event_embedding2.json, embed_model TransE, aggregate max, ...)
    args.save_path = f'models/{args.prefix}_seed{args.seed}'
    logging.info('[DS2-TRAIN] save_path (relative to EviDDIE/): %s', args.save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # module globals the trainer's loop expects from its __main__ block
    eviddie_trainer.args = args
    eviddie_trainer.loss_fn = eviddie_trainer.EvidentialLoss(annealing_step=10000)

    logging.info('[DS2-TRAIN] device: %s', eviddie_trainer.DEVICE)
    os.chdir(EVIDDIE)  # stock entry runs from EviDDIE/ (relative paths: models/, ...)
    trainer = eviddie_trainer.Trainer(args)
    logging.info('[DS2-TRAIN] dev rows loaded for checkpoint selection: %d',
                 len(trainer.dev_rows) // 2)
    trainer.train()

    import glob
    ckpts = sorted(glob.glob(os.path.join('models', f'{args.prefix}_seed{args.seed}*')))
    logging.info('[DS2-DONE] training finished; checkpoint files under EviDDIE/models/: %s', ckpts)
    logging.info('[DS2-DONE] log file: %s', os.path.join(LOG_DIR, f'log-{args.prefix}_seed{args.seed}.txt'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger().critical('DS2-TRAIN interrupted by user')
        sys.exit(130)
