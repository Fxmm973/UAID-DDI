# -*- coding: utf-8 -*-
import logging, sys, os
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import torch
import numpy as np

from eviddie_args import read_options
from eviddie_export_zs_v2 import ExportVariants

args = read_options()
args.dataset = 'dataset1'
args.no_meta = False
args.random_embed = False
args.save_dir = 'models/dataset1'
args.pretrained_model = 'models/dataset1/eviddie_0shot_seed19940419/bestmodel'
args.g_model_path = 'models/dataset1/eviddie_0shot_seed19940419/bestmodel_G'

ex = ExportVariants(args)
print('matcher.fc.5.weight shape:', tuple(ex.matcher.fc[0].weight.shape) if hasattr(ex.matcher.fc, '__getitem__') else '?')
w = ex.matcher.fc.state_dict().get('5.weight')
b = ex.matcher.fc.state_dict().get('5.bias')
print('loaded fc.5.weight mean_abs:', float(w.float().abs().mean()) if w is not None else None)
print('loaded fc.5.bias:', b.tolist() if b is not None else None)

v = ex.task_ebmedding[ex.task2id[list(ex.task2id.keys())[0]]]
g = ex.G_m(v)
print('G_m output shape:', tuple(g.shape), 'norm:', float(g.norm()), 'abs_mean:', float(g.abs().mean()))

import json
from eviddie_dataloader import DrugDataset, DrugDataLoader
rel2id = json.load(open('dataset1/relation2ids'))
tasks = json.load(open('dataset1/test2_tasks.json'))
evt = list(tasks.keys())[0]
triples = tasks[evt][:8]
rel_triples = [[t[0], t[2], rel2id[t[1]]] for t in triples]
ds = DrugDataset(rel_triples)
dl = DrugDataLoader(ds, batch_size=len(rel_triples), shuffle=False)
qb = [t.to(ex.device) for t in next(iter(dl))]
with torch.no_grad():
    ql_, qr_ = ex.matcher.model(qb)
    qn = torch.cat((ql_, qr_), dim=-1)
    y, zm, zv, zq = ex.matcher.vaemodel(qn, is_support=False, is_eval=True)
    print('model out l/r norm:', float(ql_.norm()), float(qr_.norm()))
    print('z_mean abs_mean:', float(zm.abs().mean()), 'zq abs_mean:', float(zq.abs().mean()))
    fc_out = ex.matcher.fc(torch.abs(g.expand_as(zq) - zq))
    alpha = torch.nn.functional.softplus(fc_out) + 1
    p = alpha[:, 1] / alpha.sum(1)
    print('fc_out[:3]:', fc_out[:3].tolist())
    print('p[:3]:', p[:3].tolist())
