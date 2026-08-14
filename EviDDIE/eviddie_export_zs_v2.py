#!/usr/bin/env Python
# coding=utf-8
"""
导出 EviDDIE zero-shot 消融变体预测（v2：使用固定负样本 manifest）
改动：
- 从预生成的 manifest JSON 读取负样本，不再动态采样
- SEEDS 扩展到 5 个
- 负样本固定后跨方法一致
"""
import sys
# 旧模块名 → 新模块名映射（兼容旧 checkpoint 的 pickle 路径）
import eviddie_matcher
import eviddie_modules
import eviddie_models
import eviddie_layers
_OLD_TO_NEW = {
    'matcher_structure_acc_fp_neigh_VAE_GAN_struc': eviddie_matcher,
    'matcher_structure_acc_fp_neigh_VAE_GAN_struc_ttt': eviddie_matcher,
    'modules_structure_fp_neigh': eviddie_modules,
    'models_t_struc': eviddie_models,
    'models_t_struc_ttt': eviddie_models,
    'layers': eviddie_layers,
}
for _old, _new in _OLD_TO_NEW.items():
    sys.modules[_old] = _new

import torch.nn.functional as F
import csv
from torch.autograd import Variable
from eviddie_args import read_options
from eviddie_dataloader import *
from eviddie_matcher import EmbedMatcher
from shared.checkpoint import load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

METHOD_MAP = {'softmax': 'Softmax baseline', 'evi_no_evi': 'EviDDIE w/o EVI', 'full_evi': 'EviDDIE'}
SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]  # 扩展到5个


def load_neg_manifest(dataset, split, seed):
    """Load pre-generated negative manifest."""
    path = f'neg_manifests/{split}_seed{seed}_negatives.json'
    with open(path) as f:
        return json.load(f)


class ExportVariants(object):
    def __init__(self, arg):
        for k, v in vars(arg).items(): setattr(self, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device}")

        self.semantic_task = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        for task in list(self.semantic_task.keys()):
            self.semantic_task[task] = np.array(self.semantic_task[task]) + \
                0.3*np.random.normal(0,1,size=(len(self.semantic_task[task]),1))
        self.task_ebmedding = []
        self.task2id = {}
        for num,i in enumerate(list(self.semantic_task.keys())):
            self.task2id[i]=num; self.task_ebmedding.append(self.semantic_task[i])
        self.task_ebmedding = torch.tensor(np.vstack(self.task_ebmedding)).float().to(self.device)

        self.use_pretrain = not self.random_embed
        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys())-1
        self.pad_id = self.num_symbols

        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols,
                                     use_pretrain=self.use_pretrain, embed=self.symbol2vec,
                                     dropout=self.dropout, batch_size=self.batch_size,
                                     finetune=self.fine_tune, aggregate=self.aggregate,
                                     task_emb=self.task_ebmedding).to(self.device)
        self.matcher.eval()

        ckpt = torch.load(arg.pretrained_model, map_location=self.device)
        if 'fc.5.weight' in ckpt and ckpt['fc.5.weight'].shape[0] == 1:
            logging.info('Converting old 1-output fc to 2-output...')
            ow, ob = ckpt['fc.5.weight'], ckpt['fc.5.bias']
            ckpt['fc.5.weight'] = torch.cat([ow, -ow], dim=0)
            ckpt['fc.5.bias'] = torch.cat([ob, -ob], dim=0)
        for k in list(ckpt.keys()):
            if any(x in k for x in ['symbol_emb','gcn_w','gcn_b','Bilinear','Linear_self',
                                     'Linear_nei','Linear_weak_rel','NeighborAggregator','siamese',
                                     'support_encoder','query_encoder']):
                del ckpt[k]
        load_state_dict_safe(self.matcher, ckpt, model_name='matcher')

        self.G_m = torch.load(arg.g_model_path, map_location=self.device)
        self.G_m.eval()

        self.ent2id = json.load(open(self.dataset+'/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2candidates = json.load(open(self.dataset+'/rel2candidates.json'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset+'/e1rel_e2.json'))

        # ---- 加载负样本 manifest（关键改动）----
        self.neg_manifests = {}
        for split in ['dev', 'test', 'test2']:
            self.neg_manifests[split] = {}
            for seed in SEEDS:
                self.neg_manifests[split][seed] = load_neg_manifest(arg.dataset, split, seed)

    def load_embed(self):
        symbol_id={}; rel2id=json.load(open(self.dataset+'/relation2ids'))
        ent2id=json.load(open(self.dataset+'/ent2ids'))
        r2e=json.load(open(self.dataset+'/relation2embids'))
        e2e=json.load(open(self.dataset+'/ent2embids'))
        ee=np.load(self.dataset+'/DRKG_TransE_entity.npy')
        re=np.load(self.dataset+'/DRKG_TransE_relation.npy')
        i=0; emb=[]
        for k in rel2id:
            if k not in ['','OOV']: symbol_id[k]=i; i+=1; emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        for k in ent2id:
            if k not in ['','OOV']: symbol_id[k]=i; i+=1; emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        symbol_id['PAD']=i; emb.append(list(np.zeros((re.shape[1],))))
        self.symbol2id=symbol_id; self.symbol2vec=np.array(emb)

    def build_connection(self, max_=100):
        self.connections=(np.ones((self.num_ents,max_,2))*self.pad_id).astype(int)
        self.e1_rele2=defaultdict(list); self.e1_degrees=defaultdict(int)
        with open(self.dataset+'/path_graph') as f:
            for line in tqdm(f.readlines(),desc='Connections'):
                e1,rel,e2=line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel],self.symbol2id[e2]))
        for ent,id_ in self.ent2id.items():
            nb=self.e1_rele2[ent]
            if len(nb)>max_: random.shuffle(nb); nb=nb[:max_]
            self.e1_degrees[id_]=len(nb)
            for idx,_ in enumerate(nb): self.connections[id_,idx,0]=_[0]; self.connections[id_,idx,1]=_[1]

    def get_meta(self, left, right):
        lc=Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left],axis=0))).to(self.device)
        ld=Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        rc=Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right],axis=0))).to(self.device)
        rd=Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (lc,ld,rc,rd)

    def load_head(self, variant):
        path = f'{self.save_dir}/fc_{variant}.pt'
        self.matcher.fc.load_state_dict(torch.load(path, map_location=self.device))
        self.matcher.eval()
        logging.info(f'Loaded fc head: {variant}')

    def export(self, mode, csv_writer, train_seed, eval_seed, method_name, variant):
        setting_map = {'dev':'common','test':'fewer','test2':'rare'}
        setting = setting_map.get(mode, mode)
        logging.info(f'[{method_name}] {mode.upper()} (train_seed={train_seed})')

        if mode=='dev': test_tasks=json.load(open(self.dataset+'/dev_tasks.json'))
        elif mode=='test': test_tasks=json.load(open(self.dataset+'/test_tasks.json'))
        else: test_tasks=json.load(open(self.dataset+'/test2_tasks.json'))
        rel2id=json.load(open(self.dataset+'/relation2ids'))

        # 读取预生成的固定负样本
        neg_manifest = self.neg_manifests[mode][eval_seed]

        with torch.no_grad():
            for query_ in test_tasks.keys():
                query_triples = test_tasks[query_][0:]  # few=0 for zero-shot
                if not query_triples: continue

                # 从固定 manifest 读取负样本（不再动态采样）
                manifest_entries = neg_manifest.get(query_, [])
                false_triples = []
                for entry in manifest_entries:
                    d_i, d_j, d_k, rel = entry
                    false_triples.append([d_i, rel, d_k])

                # 确保数量匹配
                if len(false_triples) != len(query_triples):
                    logging.warning(f'{query_}: manifest has {len(false_triples)} negs but {len(query_triples)} queries, skipping')
                    continue

                all_triples = query_triples + false_triples
                all_rel2id = [[t[0],t[2],rel2id[t[1]]] for t in all_triples]
                q_left = [self.ent2id[t[0]] for t in all_triples]
                q_right = [self.ent2id[t[2]] for t in all_triples]
                q_meta = self.get_meta(q_left, q_right)
                n_pos = len(query_triples)

                qb = DrugDataset(all_rel2id)
                qbl = DrugDataLoader(qb, batch_size=len(all_rel2id), shuffle=False)
                qb_data = [t.to(self.device) for t in next(iter(qbl))]
                task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()

                ql_, qr_ = self.matcher.model(qb_data)
                qn = torch.cat((ql_, qr_), dim=-1)
                _, _, _, zq = self.matcher.vaemodel(qn, is_support=False, is_eval=True)
                fc_out = self.matcher.fc(torch.abs(task_emb.expand_as(zq) - zq))

                if variant == 'softmax':
                    probs = F.softmax(fc_out, dim=1)[:, 1]
                    unc = 1.0 - torch.max(F.softmax(fc_out, dim=1), dim=1)[0]
                else:
                    evidence = F.softplus(fc_out)
                    alpha = evidence + 1
                    prob = alpha / alpha.sum(dim=1, keepdim=True)
                    probs = prob[:, 1]
                    unc = 2.0 / alpha.sum(dim=1)

                probs_np = probs.cpu().numpy()
                unc_np = unc.cpu().numpy()
                gt = np.concatenate([np.ones(n_pos), np.zeros(len(all_triples)-n_pos)])

                for idx, (t, p, u) in enumerate(zip(all_triples, probs_np, unc_np)):
                    csv_writer.writerow([train_seed, eval_seed, setting, 0, method_name, query_,
                                         t[0], t[2], int(gt[idx]), 1 if p>=0.5 else 0,
                                         round(float(p),8), round(float(u),8)])


if __name__ == '__main__':
    args = read_options()
    args.dataset = 'dataset1'
    args.no_meta = False
    args.random_embed = False
    args.save_dir = 'models/dataset1'

    TRAINING_SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
    EVAL_MANIFEST_SEED = 19940419  # 固定负样本种子
    MODES = ['dev', 'test', 'test2']

    out_dir = 'results/predictions'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'predictions_dataset1_zero_shot_variants.csv')

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['train_seed','eval_seed','setting','shot','method','event_type','drug_a','drug_b',
                     'y_true','y_pred','prob','uncertainty'])

        for train_seed in TRAINING_SEEDS:
            args.pretrained_model = f'models/dataset1/eviddie_0shot_seed{train_seed}/bestmodel'
            args.g_model_path = f'models/dataset1/eviddie_0shot_seed{train_seed}/bestmodel_G'
            if not os.path.exists(args.pretrained_model):
                logging.warning(f'Checkpoint not found: {args.pretrained_model}, skipping')
                continue

            eval_seed = EVAL_MANIFEST_SEED
            random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(eval_seed)

            for variant in ['softmax', 'evi_no_evi', 'full_evi']:
                method = METHOD_MAP[variant]
                logging.info(f'===== train_seed={train_seed} {method} =====')
                ex = ExportVariants(args)
                ex.load_head(variant)
                for mode in MODES:
                    ex.export(mode, w, train_seed, eval_seed, method, variant)

    logging.info(f'Done! Saved to {out_csv}')
    logging.info(f'Training seeds: {TRAINING_SEEDS}, Fixed eval manifest: {EVAL_MANIFEST_SEED}')
