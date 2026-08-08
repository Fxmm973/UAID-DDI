"""直接用 export_zero_shot_variants.py 的逻辑验证 checkpoint — 不做任何自定义"""
import json, logging, numpy as np, torch, torch.nn.functional as F
import os
from collections import defaultdict
from eviddie_args import read_options
from eviddie_dataloader import DrugDataset, DrugDataLoader
from eviddie_matcher import EmbedMatcher, Generate_Model
from sklearn import metrics
from shared.checkpoint import load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VerifyCheckpoint:
    """完全复刻 export_zero_shot_variants.py 的 ExportVariants 初始化逻辑"""
    def __init__(self):
        arg = read_options()
        arg.dataset = 'dataset1'; arg.semantic = 'event_embedding2.json'
        arg.embed_dim = 128; arg.batch_size = 256; arg.train_few = 10
        arg.no_meta = False; arg.random_embed = False
        arg.fine_tune = True; arg.aggregate = 'max'; arg.dropout = 0.2; arg.max_neighbor = 30
        args = arg

        # ---- 完全复刻 ExportVariants.__init__ ----
        self.semantic_task = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        for task in list(self.semantic_task.keys()):
            self.semantic_task[task] = np.array(self.semantic_task[task]) + \
                0.3*np.random.normal(0,1,size=(len(self.semantic_task[task]),1))
        self.task_ebmedding = []
        self.task2id = {}
        for num,i in enumerate(list(self.semantic_task.keys())):
            self.task2id[i]=num; self.task_ebmedding.append(self.semantic_task[i])
        self.task_ebmedding = torch.tensor(np.vstack(self.task_ebmedding)).float().to(device)

        # Load embed (same as original)
        sid={}; r2id=json.load(open(arg.dataset+'/relation2ids')); e2id=json.load(open(arg.dataset+'/ent2ids'))
        r2e=json.load(open(arg.dataset+'/relation2embids')); e2e=json.load(open(arg.dataset+'/ent2embids'))
        ee=np.load(arg.dataset+'/DRKG_TransE_entity.npy'); re=np.load(arg.dataset+'/DRKG_TransE_relation.npy')
        i=0; emb=[]
        for k in r2id:
            if k not in ['','OOV']: sid[k]=i; i+=1; emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        for k in e2id:
            if k not in ['','OOV']: sid[k]=i; i+=1; emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        sid['PAD']=i; emb.append(list(np.zeros((re.shape[1],))))
        self.symbol2id=sid; self.symbol2vec=np.array(emb)

        ns = len(sid)-1
        self.matcher = EmbedMatcher(128, ns, use_pretrain=True, embed=self.symbol2vec,
                                     dropout=0.2, batch_size=256, finetune=True,
                                     aggregate='max', task_emb=self.task_ebmedding).to(device)
        self.matcher.eval()

ckpt_paths = [
    'models/dataset1/ph2p0_0shot_40kbestmodel',
    'models/dataset1/ph2p1_0shot_40kbestmodel',
    'models/dataset1/bestmodels',
]
for ckpt_path in ckpt_paths:
    if not os.path.exists(ckpt_path):
        print(f'MISSING: {ckpt_path}')
        continue
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f'{ckpt_path}: keys={len(ckpt)}, fc_shape={ckpt.get("fc.5.weight", "N/A")}')
    if 'fc.5.weight' in ckpt and ckpt['fc.5.weight'].shape[0] == 1:
        ow,ob=ckpt['fc.5.weight'],ckpt['fc.5.bias']
        ckpt['fc.5.weight']=torch.cat([ow,-ow],0); ckpt['fc.5.bias']=torch.cat([ob,-ob],0)
    for k in list(ckpt.keys()):
        if any(x in k for x in ['symbol_emb','gcn_w','gcn_b','Bilinear','Linear_self',
            'Linear_nei','Linear_weak_rel','NeighborAggregator','siamese','support_encoder','query_encoder']):
            del ckpt[k]
    load_state_dict_safe(self.matcher, ckpt, model_name='matcher')

    self.G_m = Generate_Model(in_dim=self.task_ebmedding.shape[1]).to(device)
    self.G_m = torch.load('models/dataset1/bestmodels_G', map_location=device)
    self.G_m.eval()

    self.ent2id = e2id; self.num_ents = len(e2id)
    self.rel2candidates = json.load(open(arg.dataset+'/rel2candidates.json'))
    self.e1rel_e2 = defaultdict(list)
    self.e1rel_e2.update(json.load(open(arg.dataset+'/e1rel_e2.json')))
    self.rel2id = r2id

    self.splits = {
        'test': json.load(open(arg.dataset+'/test_tasks.json')),
        'test2': json.load(open(arg.dataset+'/test2_tasks.json')),
    }

    def eval_split(self, split_name):
        tasks = self.splits[split_name]
        all_p, all_l = [], []
        with torch.no_grad():
            for query_, triples in tasks.items():
                if not triples: continue
                cand = self.rel2candidates[query_]
                np.random.seed(2024)
                false_triples = []
                for t in triples:
                    e_h,rel,e_t=t[0],t[1],t[2]
                    while True:
                        noise=np.random.choice(cand)
                        if noise not in self.e1rel_e2.get(e_h+rel,[]) and noise!=e_t: break
                    false_triples.append([e_h,rel,noise])
                all_t = triples + false_triples
                all_r = [[t[0],t[2],self.rel2id[t[1]]] for t in all_t]
                n_pos = len(triples)

                qb = DrugDataset(all_r)
                qbl = DrugDataLoader(qb, batch_size=len(all_r), shuffle=False)
                qb_data = [t.to(device) for t in next(iter(qbl))]
                task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()

                ql_, qr_ = self.matcher.model(qb_data)
                qn = torch.cat((ql_, qr_), dim=-1)
                _, _, _, zq = self.matcher.vaemodel(qn, is_support=False, is_eval=True)
                fc_out = self.matcher.fc(torch.abs(task_emb.expand_as(zq) - zq))
                evidence = F.softplus(fc_out)
                alpha = evidence + 1
                prob = alpha[:, 1] / alpha.sum(dim=1)

                all_p.append(prob.cpu().numpy())
                all_l.append(np.concatenate([np.ones(n_pos), np.zeros(len(all_t)-n_pos)]))

        yp = np.concatenate(all_p); yt = np.concatenate(all_l)
        pred = (yp >= 0.5).astype(int)
        if len(np.unique(yt)) >= 2:
            return metrics.accuracy_score(yt,pred), metrics.roc_auc_score(yt,yp), metrics.f1_score(yt,pred,zero_division=0), len(yp)
        return 0.5,0.5,0.5,len(yp)


if __name__=='__main__':
    vc = VerifyCheckpoint()
    print(f'\nVerification — using EXACT export script logic')
    for sn in ['test', 'test2']:
        acc,au,f1,n = vc.eval_split(sn)
        print(f'{sn}: samples={n}, ACC={acc:.4f}, AUROC={au:.4f}, F1={f1:.4f}')
        # Spot-check first 5 predictions
        if au > 0.8:
            print(f'  -> CHECKPOINT IS VALID (AUROC={au:.4f})')
        else:
            print(f'  -> CHECKPOINT APPEARS BROKEN or evaluation logic differs')
