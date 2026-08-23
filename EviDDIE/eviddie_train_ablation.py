#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
import random
import csv
from sklearn import metrics

from eviddie_args import read_options
from eviddie_trainer import Trainer, EvidentialLoss
from eviddie_dataloader import train_generate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

EVAL_EVERY = 200


def compute_metrics(probas, targets):
    if len(np.unique(targets)) < 2:
        return 0.5, 0.5, 0.5
    pred = (probas >= 0.5).astype(int)
    return (metrics.accuracy_score(targets, pred),
            metrics.roc_auc_score(targets, probas),
            metrics.f1_score(targets, pred, zero_division=0))


class AblationTrainer:

    def __init__(self, args, train_seed, prefix):
        args.seed = train_seed
        args.prefix = prefix
        args.save_path = f'models/{args.prefix}_seed{args.seed}'
        self.args = args
        self.prefix = prefix
        self.train_seed = train_seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.t = Trainer(args)
        self.matcher = self.t.matcher
        self.G_m = self.t.G_m

        matcher_path = f'{args.save_path}bestmodel'
        g_path = f'{args.save_path}bestmodel_G'
        if not os.path.exists(matcher_path) or not os.path.exists(g_path):
            raise FileNotFoundError(
                f'Per-seed checkpoints not found: {matcher_path} / {g_path}. '
                f'Ablation requires the corresponding training-seed checkpoints.')
        ckpt = torch.load(matcher_path, map_location=self.device)
        self.matcher.load_state_dict(ckpt, strict=False)
        gm = torch.load(g_path, map_location=self.device)
        if isinstance(gm, dict):
            self.G_m.load_state_dict(gm)
        else:
            self.G_m.load_state_dict(gm.state_dict())
        logging.info(f'Loaded matcher ({len(ckpt)} keys) + G_m from {args.save_path}')

        for p in self.matcher.parameters():
            p.requires_grad = False
        for p in self.matcher.fc.parameters():
            p.requires_grad = True
        for p in self.G_m.parameters():
            p.requires_grad = False
        self.matcher.eval()
        self.G_m.eval()

        self.linear_proj = nn.Linear(self.t.task_ebmedding.shape[1], 64).to(self.device)

    def reset_head(self):
        for layer in self.matcher.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.linear_proj.reset_parameters()

    def _kg_encode(self, qb_data, left_ids, right_ids):
        ql_, qr_ = self.matcher.model(qb_data)
        q_meta = self.t.get_meta(left_ids, right_ids)
        ql = self.matcher.neighbor_encoder(q_meta[0], q_meta[1], ql_, qr_ - ql_)
        qr = self.matcher.neighbor_encoder(q_meta[2], q_meta[3], qr_, qr_ - ql_)
        qn = torch.cat((ql, qr), dim=-1)
        _, _, _, zq = self.matcher.vaemodel(qn, is_support=False, is_eval=True)
        return zq

    def _proto(self, task_name, use_bsa):
        sem = self.t.task_ebmedding[self.t.task2id[task_name]].unsqueeze(0)
        if use_bsa:
            return self.G_m(sem).detach()
        return self.linear_proj(sem)

    def train_variant(self, variant_name, csv_writer, max_iter):
        use_bsa = (variant_name != 'wo_BSA')
        logging.info(f'===== {variant_name} ({max_iter} iters, seed={self.train_seed}) =====')
        params = list(self.matcher.fc.parameters())
        if not use_bsa:
            params += list(self.linear_proj.parameters())
        optimizer = optim.Adam(params, lr=0.001, weight_decay=0.0)
        losses = deque([], 50)
        step = 0

        for data in train_generate(self.t.dataset, self.t.batch_size, self.t.train_few,
                                   self.t.symbol2id, self.t.ent2id, self.t.e1rel_e2,
                                   self.t.all_drug_data, self.t.drug_num_node_indices):
            task_name, _, query, false = data[0], data[1], data[2], data[3]
            ql, qr, fl, fr = data[6], data[7], data[8], data[9]
            qb, fb = data[11], data[12]
            qb = [t_.to(self.device) for t_ in qb]
            fb = [t_.to(self.device) for t_ in fb]

            proto = self._proto(task_name, use_bsa)
            self.matcher.fc.train()
            if not use_bsa:
                self.linear_proj.train()

            zq = self._kg_encode(qb, ql, qr)
            zf = self._kg_encode(fb, fl, fr)
            q_out = self.matcher.fc(torch.abs(proto.expand_as(zq) - zq))
            f_out = self.matcher.fc(torch.abs(proto.expand_as(zf) - zf))

            if variant_name == 'softmax':
                loss = F.cross_entropy(q_out, torch.ones(q_out.size(0), dtype=torch.long, device=self.device)) + \
                       F.cross_entropy(f_out, torch.zeros(f_out.size(0), dtype=torch.long, device=self.device))
            elif variant_name in ('evi_no_evi', 'wo_BSA'):
                al_q = F.softplus(q_out) + 1
                al_f = F.softplus(f_out) + 1
                loss = F.mse_loss(al_q[:, 1] / al_q.sum(1), torch.ones_like(al_q[:, 1])) + \
                       F.mse_loss(al_f[:, 1] / al_f.sum(1), torch.zeros_like(al_f[:, 1]))
            elif variant_name == 'evi_full':
                if not hasattr(self, 'edl_loss_fn'):
                    self.edl_loss_fn = EvidentialLoss(annealing_step=500)
                al_q = F.softplus(q_out) + 1
                al_f = F.softplus(f_out) + 1
                loss = self.edl_loss_fn(al_q, 'pos', step) + \
                       self.edl_loss_fn(al_f, 'neg', step)
            else:
                raise ValueError(variant_name)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == 1:
                self.matcher.fc.eval()
                if not use_bsa:
                    self.linear_proj.eval()
                acc, auroc, f1 = self._eval_dev(variant_name, use_bsa)
                self.matcher.fc.train()
                if not use_bsa:
                    self.linear_proj.train()
                logging.info(f'  [{variant_name}] step={step}/{max_iter} loss={np.mean(losses):.4f} '
                             f'auroc={auroc:.4f} f1={f1:.4f} acc={acc:.4f}')
                csv_writer.writerow([variant_name, step, np.mean(losses), auroc, f1, acc])

            if step >= max_iter:
                break

        save_dir = f'models/ablation_{self.prefix}_seed{self.train_seed}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.matcher.fc.state_dict(), os.path.join(save_dir, f'fc_{variant_name}.pt'))
        if not use_bsa:
            torch.save(self.linear_proj.state_dict(), os.path.join(save_dir, 'linear_proj_wo_BSA.pt'))
        logging.info(f'Saved {variant_name} -> {save_dir}')

    def _eval_dev(self, variant_name, use_bsa):
        rel2id = json.load(open(self.t.dataset + '/relation2ids'))
        rows_by_event = {}
        for (evt, head, rel, tail, lab) in self.t.dev_rows:
            rows_by_event.setdefault(evt, []).append((head, rel, tail, lab))
        probas, labels = [], []
        with torch.no_grad():
            for evt, ev_rows in rows_by_event.items():
                triples = [[h, t, rel2id[r]] for (h, r, t, _) in ev_rows]
                labels_e = [lab for (_, _, _, lab) in ev_rows]
                for i in range(0, len(triples), self.t.batch_size * 20):
                    batch = triples[i:i + self.t.batch_size * 20]
                    batch_labels = labels_e[i:i + self.t.batch_size * 20]
                    qb = [t_.to(self.device) for t_ in self._make_batch(batch)]
                    ql = [self.t.ent2id[t[0]] for t in batch]
                    qr = [self.t.ent2id[t[1]] for t in batch]
                    proto = self._proto(evt, use_bsa)
                    zq = self._kg_encode(qb, ql, qr)
                    fc_out = self.matcher.fc(torch.abs(proto.expand_as(zq) - zq))
                    if variant_name == 'softmax':
                        prob = F.softmax(fc_out, dim=1)[:, 1]
                    else:
                        al = F.softplus(fc_out) + 1
                        prob = al[:, 1] / al.sum(1)
                    probas.append(prob.cpu().numpy())
                    labels.append(np.asarray(batch_labels))
        if not probas:
            return 0.5, 0.5, 0.5
        return compute_metrics(np.concatenate(probas), np.concatenate(labels))

    def _make_batch(self, triples_sym):
        from eviddie_dataloader import DrugDataset, DrugDataLoader
        data = DrugDataset(triples_sym)
        loader = DrugDataLoader(data, batch_size=len(triples_sym), shuffle=False)
        return next(iter(loader))


if __name__ == '__main__':
    args = read_options()
    train_seed = int(getattr(args, 'train_seed', args.seed))
    prefix = getattr(args, 'prefix', 'eviddie_new_s1')
    variants = [v.strip() for v in getattr(args, 'variants', 'softmax,evi_no_evi,wo_BSA').split(',') if v.strip()]
    max_iter = int(getattr(args, 'max_iter', 5000))

    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    os.makedirs('results', exist_ok=True)
    csv_path = f'results/ablation_curves_{prefix}_seed{train_seed}.csv'
    header = ['variant', 'iter', 'train_loss', 'dev_auroc', 'dev_f1', 'dev_acc']

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as rf:
            reader = csv.reader(rf)
            old_header = next(reader, None)
            if old_header == header:
                existing_rows = [row for row in reader]
                logging.info(f'Loaded {len(existing_rows)} existing rows from {csv_path}')
            else:
                logging.warning(f'Header mismatch in {csv_path}; starting fresh')

    f = open(csv_path, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(header)
    for row in existing_rows:
        w.writerow(row)
    f.flush()

    tr = AblationTrainer(args, train_seed, prefix)
    for variant in variants:
        tr.reset_head()
        tr.train_variant(variant, w, max_iter)
        f.flush()

    f.close()
    logging.info(f'Done -> {csv_path}')
