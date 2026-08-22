#!/usr/bin/env Python
# coding=utf-8
import json, logging, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, os, csv
from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm
from eviddie_args import read_options
from eviddie_dataloader import *
from eviddie_matcher import EmbedMatcher
from sklearn import metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
METHODS = {'softmax':'Softmax baseline','wo_evi':'EviDDIE w/o EVI','full_evi':'EviDDIE'}

class ExportEviDDIE(object):
    def __init__(self, arg, variant):
        for k,v in vars(arg).items(): setattr(self,k,v)
        self.variant = variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device} | {METHODS[variant]}")

        self.semantic = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        for t in list(self.semantic.keys()):
            self.semantic[t] = np.array(self.semantic[t])
        self.task_emb_list=[]; self.task2id={}
        for n,i in enumerate(list(self.semantic.keys())):
            self.task2id[i]=n; self.task_emb_list.append(self.semantic[i])
        self.task_emb = torch.tensor(np.vstack(self.task_emb_list)).float().to(self.device)

        self.task_proj = nn.Sequential(
            nn.Linear(self.task_emb.shape[1],128),nn.ReLU(),nn.Linear(128,64)
        ).to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(320,256),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(256,64),nn.ReLU(),nn.Linear(64,2)
        ).to(self.device)

        self.load_embed()
        self.num_sym=len(self.symbol2id.keys())-1; self.pad_id=self.num_sym

        self.matcher = EmbedMatcher(self.embed_dim, self.num_sym,
            use_pretrain=not self.random_embed, embed=self.symbol2vec,
            dropout=self.dropout, batch_size=self.batch_size,
            finetune=self.fine_tune, aggregate=self.aggregate,
            task_emb=self.task_emb).to(self.device)
        self.matcher.eval(); self.task_proj.eval(); self.classifier.eval()

        ckpt = torch.load(f'{self.save_dir}/{variant}_model.pt', map_location=self.device)
        self.matcher.load_state_dict(ckpt['matcher'])
        self.task_proj.load_state_dict(ckpt['task_proj'])
        self.classifier.load_state_dict(ckpt['classifier'])
        logging.info(f'Loaded {METHODS[variant]}')

        self.ent2id=json.load(open(self.dataset+'/ent2ids'))
        self.num_ents=len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2c=json.load(open(self.dataset+'/rel2candidates.json'))
        self.e1rel_e2=defaultdict(list)
        self.e1rel_e2.update(json.load(open(self.dataset+'/e1rel_e2.json')))

    def load_embed(self):
        sid={};r2=json.load(open(self.dataset+'/relation2ids'))
        e2=json.load(open(self.dataset+'/ent2ids'))
        r2e=json.load(open(self.dataset+'/relation2embids'))
        e2e=json.load(open(self.dataset+'/ent2embids'))
        ee=np.load(self.dataset+'/DRKG_TransE_entity.npy');re=np.load(self.dataset+'/DRKG_TransE_relation.npy')
        i=0;emb=[]
        for k in r2:
            if k not in ['','OOV']:sid[k]=i;i+=1
            emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        for k in e2:
            if k not in ['','OOV']:sid[k]=i;i+=1
            emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        sid['PAD']=i;emb.append(list(np.zeros((re.shape[1],))))
        self.symbol2id=sid;self.symbol2vec=np.array(emb)

    def build_connection(self, max_=100):
        self.conn=(np.ones((self.num_ents,max_,2))*self.pad_id).astype(int)
        self.e1r2=defaultdict(list);self.e1d=defaultdict(int)
        with open(self.dataset+'/path_graph') as f:
            for l in tqdm(f.readlines(),desc='Connections'):
                e1,rel,e2=l.rstrip().split('\t')
                self.e1r2[e1[-7:]].append((self.symbol2id[rel],self.symbol2id[e2]))
        for ent,id_ in self.ent2id.items():
            nb=self.e1r2.get(ent, [])
            if len(nb)>max_:random.shuffle(nb);nb=nb[:max_]
            self.e1d[id_]=len(nb)
            for idx,_ in enumerate(nb):self.conn[id_,idx,0]=_[0];self.conn[id_,idx,1]=_[1]

    def get_meta(self,left,right):
        lc=Variable(torch.LongTensor(np.stack([self.conn[_,:,:] for _ in left],axis=0))).to(self.device)
        ld=Variable(torch.FloatTensor([self.e1d[_] for _ in left])).to(self.device)
        rc=Variable(torch.LongTensor(np.stack([self.conn[_,:,:] for _ in right],axis=0))).to(self.device)
        rd=Variable(torch.FloatTensor([self.e1d[_] for _ in right])).to(self.device)
        return(lc,ld,rc,rd)

    def export(self,mode,csv_writer,seed):
        mn=METHODS[self.variant]
        sm={'dev':'common','test':'fewer','test2':'rare'};setting=sm.get(mode,mode)
        logging.info(f'[{mn}] {mode.upper()}')
        if mode=='dev':tasks=json.load(open(self.dataset+'/dev_tasks.json'))
        elif mode=='test':tasks=json.load(open(self.dataset+'/test_tasks.json'))
        else:tasks=json.load(open(self.dataset+'/test2_tasks.json'))
        rel2id=json.load(open(self.dataset+'/relation2ids'))

        with torch.no_grad():
            for query_ in tasks.keys():
                candidates=self.rel2c[query_]
                qt=tasks[query_][0:];
                if not qt:continue
                ft=[]
                np.random.seed(seed+sum(ord(c) for c in query_)%10000)
                for t in qt:
                    e_h,rel,e_t=t[0],t[1],t[2]
                    while True:
                        noise=np.random.choice(candidates)
                        if (noise not in self.e1rel_e2.get(e_h+rel,[])) and noise!=e_t:break
                    ft.append([e_h,rel,noise])
                at=qt+ft;ar=[[t[0],t[2],rel2id[t[1]]] for t in at]
                n_pos=len(qt)
                qb=DrugDataset(ar);qbl=DrugDataLoader(qb,batch_size=len(ar),shuffle=False)
                qb_data=[t.to(self.device) for t in next(iter(qbl))]
                tp=self.task_proj(self.task_emb[self.task2id[query_]])
                ql_,qr_=self.matcher.model(qb_data)
                feat=torch.cat([ql_,qr_,tp.expand(ql_.shape[0],-1)],dim=-1)
                fc_out=self.classifier(feat)
                if self.variant=='softmax':
                    probs=F.softmax(fc_out,dim=1)[:,1]
                    unc=1.0-torch.max(F.softmax(fc_out,dim=1),dim=1)[0]
                else:
                    ev=F.softplus(fc_out);alpha=ev+1
                    prob=alpha/alpha.sum(dim=1,keepdim=True)
                    probs=prob[:,1];unc=2.0/alpha.sum(dim=1)
                pn=probs.cpu().numpy();un=unc.cpu().numpy()
                gt=np.concatenate([np.ones(n_pos),np.zeros(len(at)-n_pos)])
                for idx,(t,p,u) in enumerate(zip(at,pn,un)):
                    csv_writer.writerow([seed,setting,0,mn,query_,
                        t[0],t[2],int(gt[idx]),1 if p>=0.5 else 0,
                        round(float(p),8),round(float(u),8)])

if __name__=='__main__':
    args=read_options()
    args.dataset='dataset1';args.no_meta=False;args.random_embed=False
    args.save_dir='models/dataset1/zero_shot_variants'
    SEEDS=[2024,2025,2026];MODES=['dev','test','test2']
    out_dir='results/predictions';os.makedirs(out_dir,exist_ok=True)
    out_csv=os.path.join(out_dir,'predictions_dataset1_zero_shot_variants.csv')
    with open(out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['seed','setting','shot','method','event_type','drug_a','drug_b',
                     'y_true','y_pred','prob','uncertainty'])
        for variant in ['softmax','wo_evi','full_evi']:
            ex=ExportEviDDIE(args,variant)
            for seed in SEEDS:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
                for mode in MODES:ex.export(mode,w,seed)
    logging.info(f'Done! Saved to {out_csv}')
