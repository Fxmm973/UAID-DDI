#!/usr/bin/env Python
# coding=utf-8
"""
EviDDIE 三变体训练 (简化版)：CSE+VAE+fc 联合训练，无 GAN。
softmax / w/o EVI / full EVI 各 10000 iter (~15 min each)
"""
import json, logging, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, os, sys
from collections import defaultdict, deque
from torch import optim
from tqdm import tqdm
from eviddie_args import read_options
from eviddie_dataloader import *
from eviddie_matcher import EmbedMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def edl_loss(alpha, y_true, anneal=0.005):
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = (p[:,1] - y_true)**2
    var = (p[:,1] * (1 - p[:,1])) / (S.squeeze() + 1)
    mse = (err + var).mean()
    K = alpha.shape[1]; beta = torch.ones_like(alpha)
    kl = torch.lgamma(S) - torch.lgamma(torch.tensor(K, dtype=S.dtype, device=S.device)) \
         - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=1) \
         + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return mse + anneal * kl.mean(), mse, kl.mean()

class Trainer(object):
    def __init__(self, arg):
        for k,v in vars(arg).items(): setattr(self,k,v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device}")

        # Task embeddings
        self.semantic = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        for t in list(self.semantic.keys()):
            self.semantic[t] = np.array(self.semantic[t]) + 0.3*np.random.normal(0,1,size=(len(self.semantic[t]),1))
        self.task_emb_list = []; self.task2id = {}
        for n,i in enumerate(list(self.semantic.keys())):
            self.task2id[i]=n; self.task_emb_list.append(self.semantic[i])
        self.task_emb = torch.tensor(np.vstack(self.task_emb_list)).float().to(self.device)

        # Simple task projector (replaces G_m)
        self.task_proj = nn.Sequential(
            nn.Linear(self.task_emb.shape[1], 128), nn.ReLU(), nn.Linear(128, 64)
        ).to(self.device)

        # Embeddings
        self.load_embed()
        self.num_sym = len(self.symbol2id.keys())-1; self.pad_id=self.num_sym

        # Encoder (we'll use model + VAE directly, create our own classifier)
        self.matcher = EmbedMatcher(self.embed_dim, self.num_sym, use_pretrain=not self.random_embed,
                                     embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
                                     finetune=self.fine_tune, aggregate=self.aggregate,
                                     task_emb=self.task_emb).to(self.device)

        # Classifier: CSE(256) + task_proj(64) = 320-dim input
        self.classifier = nn.Sequential(
            nn.Linear(320, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2)
        ).to(self.device)

        # Data
        self.ent2id = json.load(open(self.dataset+'/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2c = json.load(open(self.dataset+'/rel2candidates.json'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2.update(json.load(open(self.dataset+'/e1rel_e2.json')))
        self.all_dd = {}; self.drug_ni = {}

    def load_embed(self):
        sid={}; r2=json.load(open(self.dataset+'/relation2ids'))
        e2=json.load(open(self.dataset+'/ent2ids'))
        r2e=json.load(open(self.dataset+'/relation2embids'))
        e2e=json.load(open(self.dataset+'/ent2embids'))
        ee=np.load(self.dataset+'/DRKG_TransE_entity.npy'); re=np.load(self.dataset+'/DRKG_TransE_relation.npy')
        i=0; emb=[]
        for k in r2:
            if k not in ['','OOV']: sid[k]=i; i+=1
            emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        for k in e2:
            if k not in ['','OOV']: sid[k]=i; i+=1
            emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        sid['PAD']=i; emb.append(list(np.zeros((re.shape[1],))))
        self.symbol2id=sid; self.symbol2vec=np.array(emb)

    def build_connection(self, max_=100):
        self.conn=(np.ones((self.num_ents,max_,2))*self.pad_id).astype(int)
        self.e1r2=defaultdict(list); self.e1d=defaultdict(int)
        with open(self.dataset+'/path_graph') as f:
            for l in tqdm(f.readlines(),desc='Build conn'):
                e1,rel,e2=l.rstrip().split('\t')
                self.e1r2[e1[-7:]].append((self.symbol2id[rel],self.symbol2id[e2]))
        for ent,id_ in self.ent2id.items():
            nb=self.e1r2.get(ent, [])
            if len(nb)>max_: random.shuffle(nb); nb=nb[:max_]
            self.e1d[id_]=len(nb)
            for idx,_ in enumerate(nb): self.conn[id_,idx,0]=_[0]; self.conn[id_,idx,1]=_[1]

    def train(self, variant, max_iter=10000):
        logging.info(f'=== {variant} | {max_iter} iters ===')
        all_p = list(self.matcher.parameters()) + list(self.task_proj.parameters()) + list(self.classifier.parameters())
        vae_ids = {id(p) for p in self.matcher.vaemodel.parameters()}
        main_params = [p for p in all_p if id(p) not in vae_ids]
        optimizer = optim.Adam(main_params, lr=0.001)
        opt_vae = optim.Adam(self.matcher.vaemodel.parameters(), lr=0.01)
        losses = deque([], 50); bn = 0

        for data in train_generate(self.dataset, self.batch_size, self.train_few,
                                     self.symbol2id, self.ent2id, self.e1rel_e2,
                                     self.all_dd, self.drug_ni):
            tn,_,query,false,_,_,_,_,_,_, sb,qb,fb = data
            sb=[t.to(self.device) for t in sb]; qb=[t.to(self.device) for t in qb]
            fb=[t.to(self.device) for t in fb]

            # Minimal: just CSE features → classifier (no VAE, no task_proj)
            ql_,qr_=self.matcher.model(qb); pos_feat = torch.cat((ql_,qr_), dim=-1)  # [B,256]
            fl_,fr_=self.matcher.model(fb); neg_feat = torch.cat((fl_,fr_), dim=-1)
            # Also add task embedding difference for discrimination
            tp = self.task_proj(self.task_emb[self.task2id[tn]])
            pos_feat = torch.cat([pos_feat, tp.expand(pos_feat.shape[0], -1)], dim=-1)  # [B,320]
            neg_feat = torch.cat([neg_feat, tp.expand(neg_feat.shape[0], -1)], dim=-1)

            q_out = self.classifier(pos_feat)
            f_out = self.classifier(neg_feat)

            if variant == 'softmax':
                p_l = -F.logsigmoid(q_out[:,1]-q_out[:,0]).mean()
                n_l = -F.logsigmoid(-(f_out[:,1]-f_out[:,0])).mean()
                loss = (p_l+n_l)/2
            elif variant == 'wo_evi':
                eq=F.softplus(q_out); aq=eq+1; pq=aq[:,1]/aq.sum(dim=1)
                ef=F.softplus(f_out); af=ef+1; pf=af[:,1]/af.sum(dim=1)
                loss = F.binary_cross_entropy(pq, torch.ones_like(pq)) + F.binary_cross_entropy(pf, torch.zeros_like(pf))
            else:
                eq=F.softplus(q_out); aq=eq+1
                ef=F.softplus(f_out); af=ef+1
                lq,_,_ = edl_loss(aq, torch.ones(aq.shape[0], device=self.device))
                lf,_,_ = edl_loss(af, torch.zeros(af.shape[0], device=self.device))
                loss = lq+lf

            losses.append(loss.item())
            optimizer.zero_grad(); opt_vae.zero_grad(); loss.backward()
            optimizer.step(); opt_vae.step()

            bn+=1
            if bn%200==0:
                logging.info(f'  [{variant}] {bn}/{max_iter} loss={np.mean(losses):.4f}')
            if bn>=max_iter: break

        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({'matcher':self.matcher.state_dict(),'task_proj':self.task_proj.state_dict(),
                    'classifier':self.classifier.state_dict()},
                   f'{self.save_dir}/{variant}_model.pt')
        logging.info(f'Saved {self.save_dir}/{variant}_model.pt')

if __name__=='__main__':
    args=read_options()
    args.dataset='dataset1'; args.max_batches=10000; args.batch_size=256
    args.train_few=10; args.lr=0.001; args.weight_decay=0.0; args.dropout=0.2
    args.no_meta=False; args.random_embed=False
    args.save_dir='models/dataset1/zero_shot_variants'

    for v in ['softmax','wo_evi','full_evi']:
        random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(2024)
        t=Trainer(args); t.train(v, max_iter=3000)
    logging.info('Done!')
