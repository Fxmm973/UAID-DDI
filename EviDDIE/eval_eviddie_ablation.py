#!/usr/bin/env python
# coding=utf-8
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'PharDDIE'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eviddie_matcher as _em
for _old in ['matcher_structure_acc_fp_neigh_VAE_GAN_struc',
             'matcher_structure_acc_fp_neigh_VAE_GAN_struc_ttt']:
    sys.modules[_old] = _em

from pharddie_matcher import MVN_DDI, VAE
from eviddie_matcher import Generate_Model
from eviddie_dataloader import DrugDataset, DrugDataLoader
from shared.checkpoint import load_state_dict_safe

EVAL_SEED = 19940419
VARIANTS = ['softmax', 'evi_no_evi', 'full_evi']
SPLITS = ['dev', 'test', 'test2']
CKPT = 'models/dataset1/pharddie_best.pt'
G_PATH = 'models/dataset1/bestmodels_G'
MANIFEST_DIR = 'neg_manifests'
HEADS = {v: f'models/dataset1/fc_{v}.pt' for v in VARIANTS}


def build_backbone(device):
    enc = MVN_DDI([55, 2048, 200], 17, 128, 128, 0, [64, 64], [2, 2], 64, 0.0).to(device)
    srae = VAE(emb_dim=256).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    if hasattr(ckpt, 'state_dict'):
        ckpt = ckpt.state_dict()
    enc_state = {k.replace('model.', ''): v for k, v in ckpt.items() if k.startswith('model.')}
    srae_state = {k.replace('vaemodel.', ''): v for k, v in ckpt.items() if k.startswith('vaemodel.')}
    load_state_dict_safe(enc, enc_state, model_name='backbone_encoder')
    load_state_dict_safe(srae, srae_state, model_name='backbone_srae')
    enc.eval(); srae.eval()
    for p in list(enc.parameters()) + list(srae.parameters()):
        p.requires_grad = False
    return enc, srae


def build_gm(device):
    gm = torch.load(G_PATH, map_location=device, weights_only=False)
    if isinstance(gm, dict):
        raise RuntimeError('G_m should be a full module pickle, got a state dict')
    gm = gm.to(device).eval()
    for p in gm.parameters():
        p.requires_grad = False
    return gm


def build_head(variant, device):
    fc = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 2),
    ).to(device)
    fc.load_state_dict(torch.load(HEADS[variant], map_location=device, weights_only=False))
    fc.eval()
    for p in fc.parameters():
        p.requires_grad = False
    return fc


def ece15(conf, pred, lab):
    b = np.linspace(0, 1, 16)
    e = 0.0
    for i in range(15):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum() > 0:
            e += m.sum() / len(conf) * np.abs((pred[m] == lab[m]).mean() - conf[m].mean())
    return e


def metrics_for(p, y, ev):
    pred = (p >= 0.5).astype(int)
    auc = metrics.roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    acc = metrics.accuracy_score(y, pred)
    f1s = []
    for e in np.unique(ev):
        m = ev == e
        if len(np.unique(y[m])) > 1:
            f1s.append(metrics.f1_score(y[m], pred[m], zero_division=0))
    f1 = np.mean(f1s) if f1s else float('nan')
    conf = np.maximum(p, 1 - p)
    ece = ece15(conf, pred, y)
    brier = np.mean((p - y) ** 2)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    nll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    hc = conf > 0.9
    hce = (pred[hc] != y[hc]).mean() if hc.sum() > 0 else 0.0
    return auc, acc, f1, ece, brier, nll, hce


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EVAL_SEED)

    enc, srae = build_backbone(device)
    gm = build_gm(device)
    heads = {v: build_head(v, device) for v in VARIANTS}

    task_emb = json.load(open('dataset1/event_embedding2.json'))
    task_emb = {k: np.array(v, dtype=np.float32).reshape(-1) for k, v in task_emb.items()}

    hash_log = json.load(open(os.path.join(MANIFEST_DIR, 'manifest_hashes.json')))
    results = []
    with torch.no_grad():
        for split in SPLITS:
            mf = os.path.join(MANIFEST_DIR, f'{split}_seed{EVAL_SEED}_negatives.json')
            actual = hashlib.sha256(open(mf, 'rb').read()).hexdigest()
            recorded = hash_log.get(f'{split}_seed{EVAL_SEED}', {}).get('sha256')
            if recorded is None or actual != recorded:
                raise RuntimeError(f'manifest SHA256 mismatch: {mf}')
            manifest = json.load(open(mf))
            tasks = json.load(open(f'dataset1/{split}_tasks.json'))

            acc_p = {v: [] for v in VARIANTS}
            acc_y, acc_ev = [], []
            for event in tqdm(tasks.keys(), desc=f'eval {split}'):
                triples = tasks[event]
                if not triples:
                    continue
                entries = manifest.get(event, [])
                if len(entries) != len(triples):
                    raise RuntimeError(f'{event}: manifest has {len(entries)} entries vs {len(triples)} triples')
                false_triples = []
                for t, entry in zip(triples, entries):
                    d_i, d_j, d_k, rel = entry
                    if not (d_i == t[0] and d_j == t[2] and rel == t[1]):
                        raise RuntimeError(f'manifest entry mismatch: {entry} vs {t}')
                    false_triples.append([t[0], t[1], d_k])
                all_triples = triples + false_triples
                rel2id = json.load(open('dataset1/relation2ids'))
                ar = [[t[0], t[2], rel2id[t[1]]] for t in all_triples]
                n_pos = len(triples)

                qb = DrugDataset(ar)
                qb_data = [t.to(device) for t in next(iter(DrugDataLoader(qb, batch_size=len(ar), shuffle=False)))]
                hl, hr = enc(qb_data)
                qn = torch.cat((hl, hr), dim=-1)
                _, _, _, z = srae(qn, is_support=False, is_eval=True)

                proto = gm(torch.tensor(task_emb[event], device=device).unsqueeze(0)).detach()
                for v in VARIANTS:
                    fc_out = heads[v](torch.abs(proto.expand_as(z) - z))
                    if v == 'softmax':
                        p = F.softmax(fc_out, dim=1)[:, 1]
                    else:
                        al = F.softplus(fc_out) + 1
                        p = al[:, 1] / al.sum(1)
                    acc_p[v].append(p.cpu().numpy())
                acc_y.append(np.concatenate([np.ones(n_pos), np.zeros(len(all_triples) - n_pos)]))
                acc_ev.append(np.array([event] * len(all_triples)))

            y = np.concatenate(acc_y)
            ev = np.concatenate(acc_ev)
            for v in VARIANTS:
                p = np.concatenate(acc_p[v])
                auc, acc, f1, ece, brier, nll, hce = metrics_for(p, y, ev)
                results.append([v, split, auc, acc, f1, ece, brier, nll, hce])
                print(f'[{v:10s} {split:5s}] AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f} | '
                      f'ECE={ece:.4f} Brier={brier:.4f} NLL={nll:.4f} HCE={hce:.4f}')

    os.makedirs('results', exist_ok=True)
    with open('results/eviddie_ablation_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'split', 'auc', 'acc', 'f1', 'ece', 'brier', 'nll', 'hce'])
        w.writerows(results)
    print('\nSaved to results/eviddie_ablation_results.csv')


if __name__ == '__main__':
    main()
