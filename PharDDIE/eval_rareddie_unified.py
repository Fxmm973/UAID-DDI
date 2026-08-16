#!/usr/bin/env python
# coding=utf-8
"""
eval_rareddie_unified.py — 统一协议重评 RareDDIE（在 UAID-DDI/PharDDIE 目录下运行）
协议与 PharDDIE 完全一致：
  - support = 每事件前 few 个三元组
  - 负样本 = 固定 manifest（seed 19940419，SHA256 校验）
  - 指标 = pooled AUC / ACC + event-macro F1
训练种子：19940419, 20230801, 20240115, 20240520, 20240910（与 PharDDIE 相同）
用法（每种子一条，训练完成后）：
  python eval_rareddie_unified.py --few 1 --mode test2 --seed 19940419 \
      --checkpoint models/rareddie_1shot_seed19940419bestmodel
"""
import argparse, json, hashlib, os
import numpy as np
import torch
from torch.autograd import Variable
from sklearn import metrics
from tqdm import tqdm

from matcher_structure_acc_fp_neigh_VAE_struc import EmbedMatcher
from data_loader_structure_fp import DrugDataset, DrugDataLoader

TRAIN_SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
EVAL_SEED = 19940419
MAX_NEIGHBOR = 30
EMBED_DIM = 128


class UnifiedEval(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.few = args.few
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_embed()
        self.num_symbols = len(self.symbol2id) - 1
        self.pad_id = self.num_symbols

        self.matcher = EmbedMatcher(
            EMBED_DIM, self.num_symbols,
            use_pretrain=True, embed=self.symbol2vec,
            dropout=0.2, batch_size=256,
            finetune=True, aggregate='max',
        ).to(self.device)
        ckpt = torch.load(args.checkpoint, map_location=self.device)
        if hasattr(ckpt, 'state_dict'):
            ckpt = ckpt.state_dict()
        for k in list(ckpt.keys()):
            if any(x in k for x in ['support_encoder.proj', 'support_encoder.layer_norm',
                                     'query_encoder.process']):
                del ckpt[k]
        self.matcher.load_state_dict(ckpt, strict=False)
        self.matcher.eval()
        for p in self.matcher.parameters():
            p.requires_grad = False

        self.ent2id = json.load(open(f'{self.dataset}/ent2ids'))
        self.rel2id = json.load(open(f'{self.dataset}/relation2ids'))
        self.rel2candidates = json.load(open(f'{self.dataset}/rel2candidates.json'))
        self.e1rel_e2 = json.load(open(f'{self.dataset}/e1rel_e2.json'))
        self.build_connection()

        mf = os.path.join(args.manifest_dir, f'{args.mode}_seed{EVAL_SEED}_negatives.json')
        if not os.path.exists(mf):
            raise FileNotFoundError(f'manifest 不存在: {mf}。请把 --manifest-dir 指向 '
                                    f'UAID-DDI 的 PharDDIE/dataset1/neg_manifests')
        hash_log = json.load(open(os.path.join(args.manifest_dir, 'manifest_hashes.json')))
        recorded = hash_log.get(f'{args.mode}_seed{EVAL_SEED}', {}).get('sha256')
        actual = hashlib.sha256(open(mf, 'rb').read()).hexdigest()
        if recorded is None or actual != recorded:
            raise RuntimeError(f'manifest SHA256 不匹配: {mf}')
        self.manifest = json.load(open(mf))
        print(f'[MANIFEST-CHAIN] {args.mode}_seed{EVAL_SEED} SHA256 verified.')

    def load_embed(self):
        rel2id = json.load(open(f'{self.dataset}/relation2ids'))
        ent2id = json.load(open(f'{self.dataset}/ent2ids'))
        rel2emb = json.load(open(f'{self.dataset}/relation2embids'))
        ent2emb = json.load(open(f'{self.dataset}/ent2embids'))
        ent_e = np.load(f'{self.dataset}/DRKG_TransE_entity.npy')
        rel_e = np.load(f'{self.dataset}/DRKG_TransE_relation.npy')
        symbol_id, emb = {}, []
        for k in rel2id:
            if k in ('', 'OOV'):
                continue
            symbol_id[k] = len(symbol_id)
            emb.append(rel_e[rel2emb[k], :] if rel2emb[k] != -1 else np.random.randn(rel_e.shape[1]))
        for k in ent2id:
            if k in ('', 'OOV'):
                continue
            symbol_id[k] = len(symbol_id)
            emb.append(ent_e[ent2emb[k], :] if ent2emb[k] != -1 else np.random.randn(rel_e.shape[1]))
        symbol_id['PAD'] = len(symbol_id)
        emb.append(np.zeros((rel_e.shape[1],)))
        self.symbol2id = symbol_id
        self.symbol2vec = np.array(emb)

    def build_connection(self):
        num_ents = len(self.ent2id)
        self.connections = (np.ones((num_ents, MAX_NEIGHBOR, 2)) * self.pad_id).astype(int)
        self.e1_degrees = {}
        e1_rele2 = {}
        with open(f'{self.dataset}/path_graph') as f:
            for line in tqdm(f.readlines(), desc='Building connections'):
                e1, rel, e2 = line.rstrip().split('\t')
                e1_rele2.setdefault(e1[-7:], []).append((self.symbol2id[rel], self.symbol2id[e2]))
        for ent, id_ in self.ent2id.items():
            nb = e1_rele2.get(ent, [])
            if len(nb) > MAX_NEIGHBOR:
                rng = np.random.RandomState(EVAL_SEED)
                idx = rng.choice(len(nb), MAX_NEIGHBOR, replace=False)
                nb = [nb[i] for i in idx]
            self.e1_degrees[id_] = len(nb)
            for j, (r, e) in enumerate(nb):
                self.connections[id_, j, 0] = r
                self.connections[id_, j, 1] = e

    def get_meta(self, left, right):
        lc = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], 0))).to(self.device)
        ld = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        rc = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], 0))).to(self.device)
        rd = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (lc, ld, rc, rd)

    def eval_split(self, mode):
        tasks = json.load(open(f'{self.dataset}/{mode}_tasks.json'))
        probas, labels, events = [], [], []
        with torch.no_grad():
            for query_ in tqdm(tasks.keys(), desc=f'eval {mode}'):
                triples = tasks[query_]
                if len(triples) < self.few + 1:
                    continue
                support_triples = triples[:self.few]
                query_triples = triples[self.few:]
                # ---- 负样本读固定 manifest（与 pharddie_export_full.py 逐条对齐）----
                entries = self.manifest.get(query_, [])[self.few:]
                if len(entries) != len(query_triples):
                    raise RuntimeError(f'{query_}: manifest {len(entries)} vs queries {len(query_triples)}')
                false_triples = []
                for t, entry in zip(query_triples, entries):
                    d_i, d_j, d_k, rel = entry
                    if not (d_i == t[0] and d_j == t[2] and rel == t[1]):
                        raise RuntimeError(f'manifest entry mismatch: {entry} vs {t}')
                    false_triples.append([t[0], t[1], d_k])

                all_triples = query_triples + false_triples
                all_rel2id = [[t[0], t[2], self.rel2id[t[1]]] for t in all_triples]
                n_pos = len(query_triples)

                sup_rel2id = [[t[0], t[2], self.rel2id[t[1]]] for t in support_triples]
                sb = DrugDataset(sup_rel2id)
                sb_data = [t.to(self.device) for t in next(iter(
                    DrugDataLoader(sb, batch_size=len(sup_rel2id), shuffle=False)))]
                s_left = [self.ent2id[t[0]] for t in support_triples]
                s_right = [self.ent2id[t[2]] for t in support_triples]
                s_meta = self.get_meta(s_left, s_right)
                support_pairs = [[self.symbol2id[t[0]], self.symbol2id[t[2]]] for t in support_triples]

                qb = DrugDataset(all_rel2id)
                qb_data = [t.to(self.device) for t in next(iter(
                    DrugDataLoader(qb, batch_size=len(all_rel2id), shuffle=False)))]
                q_left = [self.ent2id[t[0]] for t in all_triples]
                q_right = [self.ent2id[t[2]] for t in all_triples]
                q_meta = self.get_meta(q_left, q_right)
                query_pairs = [[self.symbol2id[t[0]], self.symbol2id[t[2]]] for t in all_triples]

                support = Variable(torch.LongTensor(support_pairs)).to(self.device)
                query = Variable(torch.LongTensor(query_pairs)).to(self.device)
                scores, _ = self.matcher(query, support, q_meta, s_meta, qb_data, sb_data, None)
                probs = torch.sigmoid(scores.detach().cpu()).numpy().flatten()
                probas.append(probs)
                labels.append(np.concatenate([np.ones(n_pos), np.zeros(len(all_triples) - n_pos)]))
                events.append(np.array([query_] * len(probs)))

        p = np.concatenate(probas); y = np.concatenate(labels); ev = np.concatenate(events)
        pred = (p >= 0.5).astype(int)
        auc = metrics.roc_auc_score(y, p)
        acc = metrics.accuracy_score(y, pred)
        f1s = []
        for e in np.unique(ev):
            m = ev == e
            if len(np.unique(y[m])) > 1:
                f1s.append(metrics.f1_score(y[m], pred[m], zero_division=0))
        macro_f1 = np.mean(f1s) if f1s else 0.0
        print(f'\n[UNIFIED {mode}] AUC={auc:.4f}  ACC={acc:.4f}  macro-F1={macro_f1:.4f}  (n={len(y)})')
        return auc, acc, macro_f1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset1')
    ap.add_argument('--few', type=int, default=1)
    ap.add_argument('--mode', default='test2', choices=['dev', 'test', 'test2'])
    ap.add_argument('--seed', type=int, default=19940419, help='训练种子（对应 checkpoint）')
    ap.add_argument('--checkpoint', default='models/rareddie_1shot_seed19940419bestmodel')
    ap.add_argument('--manifest-dir', default='dataset1/neg_manifests')
    args = ap.parse_args()
    assert args.seed in TRAIN_SEEDS, f'训练种子应为 PharDDIE 同款: {TRAIN_SEEDS}'
    np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
    ev = UnifiedEval(args)
    ev.eval_split(args.mode)
