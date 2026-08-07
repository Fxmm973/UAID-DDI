#!/usr/bin/env python
"""直接评估完整 EviDDIE checkpoint 在 dev set 上的表现，验证 checkpoint 是否有效"""
import json, logging, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from eviddie_args import read_options
from eviddie_dataloader import DrugDataset, DrugDataLoader
from eviddie_matcher import EmbedMatcher, Generate_Model
from sklearn import metrics
from shared.checkpoint import convert_fc_1to2, load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def comp(probas, targets):
    if len(np.unique(targets)) < 2: return 0.5, 0.5, 0.5, len(targets)
    pred = (probas >= 0.5).astype(int)
    return (metrics.accuracy_score(targets, pred),
            metrics.roc_auc_score(targets, probas),
            metrics.f1_score(targets, pred, zero_division=0),
            len(targets))


class DirectEval:
    """直接加载完整 EviDDIE checkpoint + G_m，不做任何额外训练"""
    def __init__(self):
        arg = read_options()
        arg.dataset = 'dataset1'; arg.semantic = 'event_embedding2.json'
        arg.embed_dim = 128; arg.batch_size = 256; arg.train_few = 10
        arg.no_meta = False; arg.random_embed = False
        arg.fine_tune = True; arg.aggregate = 'max'; arg.dropout = 0.2; arg.max_neighbor = 30

        sm = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        for t in sm: sm[t] = np.array(sm[t]) + 0.3*np.random.normal(0,1,size=(len(sm[t]),1))
        self.te = torch.tensor(np.vstack([sm[k] for k in sm])).float().to(device)
        self.t2id = {k:i for i,k in enumerate(sm)}  # preserve insertion order, same as export scripts

        sid={}; r2=json.load(open(arg.dataset+'/relation2ids')); e2=json.load(open(arg.dataset+'/ent2ids'))
        r2e=json.load(open(arg.dataset+'/relation2embids')); e2e=json.load(open(arg.dataset+'/ent2embids'))
        ee=np.load(arg.dataset+'/DRKG_TransE_entity.npy'); re=np.load(arg.dataset+'/DRKG_TransE_relation.npy')
        i=0; emb=[]
        for k in sorted(r2):
            if k not in ['','OOV']: sid[k]=i; i+=1; emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        for k in sorted(e2):
            if k not in ['','OOV']: sid[k]=i; i+=1; emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        sid['PAD']=i; emb.append(list(np.zeros((re.shape[1],))))
        self.sym2id=sid; self.sym2vec=np.array(emb)

        self.matcher = EmbedMatcher(128, len(sid)-1, use_pretrain=True, embed=self.sym2vec,
                                     dropout=0.2, batch_size=256, finetune=True,
                                     aggregate='max', task_emb=self.te).to(device)

        ckpt_path = 'models/dataset1/bestmodels'
        ckpt = torch.load(ckpt_path, map_location=device)
        logging.info(f'Checkpoint keys: {list(ckpt.keys())[:10]}...')
        # Check fc output dim
        if 'fc.5.weight' in ckpt:
            logging.info(f'fc.5.weight shape: {ckpt["fc.5.weight"].shape}')

convert_fc_1to2(ckpt)
        for k in list(ckpt.keys()):
            if any(x in k for x in ['symbol_emb','gcn_w','gcn_b','Bilinear','Linear_self',
                'Linear_nei','Linear_weak_rel','NeighborAggregator','siamese','support_encoder','query_encoder']):
                del ckpt[k]

        missing, unexpected = load_state_dict_safe(self.matcher, ckpt, model_name='matcher')
        logging.info(f'Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
        if len(missing) > 0:
            logging.info(f'First 5 missing: {missing[:5]}')
        self.matcher.eval()

        self.G_m = Generate_Model(in_dim=self.te.shape[1]).to(device)
        self.G_m = torch.load('models/dataset1/bestmodels_G', map_location=device)
        self.G_m.eval()

        self.ent2id=e2; self.num_ents=len(e2)
        self.rel2c=json.load(open(arg.dataset+'/rel2candidates.json'))
        self.e1re2=defaultdict(list); self.e1re2.update(json.load(open(arg.dataset+'/e1rel_e2.json')))
        self.rel2id=r2
        # Evaluate on dev + test + test2
        self.splits = {
            'dev': json.load(open(arg.dataset+'/dev_tasks.json')),
            'test': json.load(open(arg.dataset+'/test_tasks.json')),
            'test2': json.load(open(arg.dataset+'/test2_tasks.json')),
        }

    def encode(self, batch):
        ql,qr = self.matcher.model(batch)
        qn = torch.cat((ql,qr),-1)
        _,_,_,z = self.matcher.vaemodel(qn, is_support=False, is_eval=True)
        return z

    def proto(self, name):
        with torch.no_grad():
            return self.G_m(self.te[self.t2id[name]]).detach()

    def eval_split(self, split_name):
        tasks = self.splits[split_name]
        all_p, all_l = [], []
        with torch.no_grad():
            for q, triples in tasks.items():
                if not triples: continue
                cand = self.rel2c[q]
                np.random.seed(hash(q)%100000+42)
                ft=[]
                for t in triples:
                    eh,rel,et=t[0],t[1],t[2]
                    while True:
                        n=np.random.choice(cand)
                        if n not in self.e1re2.get(eh+rel,[]) and n!=et: break
                    ft.append([eh,rel,n])
                at=triples+ft; ar=[[t[0],t[2],self.rel2id[t[1]]] for t in at]; npos=len(triples)
                qb=DrugDataset(ar); qbl=DrugDataLoader(qb,batch_size=len(ar),shuffle=False)
                qbd=[t.to(device) for t in next(iter(qbl))]
                p=self.proto(q); zq=self.encode(qbd)
                out=self.matcher.fc(torch.abs(p.expand_as(zq)-zq))
                ev=F.softplus(out); al=ev+1; prob=al[:,1]/al.sum(1)
                all_p.append(prob.cpu().numpy())
                all_l.append(np.concatenate([np.ones(npos),np.zeros(len(at)-npos)]))
        return comp(np.concatenate(all_p),np.concatenate(all_l))


if __name__=='__main__':
    ev = DirectEval()
    print(f'\n{"="*70}')
    print('EviDDIE Full Model — Direct Evaluation (no retraining)')
    print(f'{"="*70}')
    print(f'{"Split":<10} {"Samples":<10} {"ACC":<10} {"AUROC":<10} {"F1":<10}')
    print('-'*50)
    for sn in ['dev', 'test', 'test2']:
        acc,au,f1,n = ev.eval_split(sn)
        print(f'{sn:<10} {n:<10} {acc:<10.4f} {au:<10.4f} {f1:<10.4f}')
