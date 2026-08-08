#!/usr/bin/env Python
# coding=utf-8
"""
EviDDIE 0-shot 评估脚本 — 导出逐样本预测 + EDL uncertainty。
修复了原 tester 的 double-sigmoid bug，正确提取 Dirichlet alpha。
"""
import csv, types
from eviddie_args import read_options
from eviddie_dataloader import *
from eviddie_matcher import *
from sklearn import metrics
from shared.checkpoint import convert_fc_1to2, load_state_dict_safe, log_seed_checkpoint_note

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred) if len(np.unique(target))>1 else 0.0
    f1 = metrics.f1_score(target, pred, zero_division=0)
    return acc, auroc, f1


class EviDDIEExport(object):
    def __init__(self, arg):
        for k, v in vars(arg).items(): setattr(self, k, v)
        self.meta = not self.no_meta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # 设置 use_pretrain（args.py 里没有这个参数）
        if self.random_embed:
            self.use_pretrain = False
        else:
            self.use_pretrain = True

        # 加载 semantic task embedding
        self.semantic_task = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        for task in list(self.semantic_task.keys()):
            self.semantic_task[task] = np.array(self.semantic_task[task]) + 0.3 * np.random.normal(loc=0, scale=1,
                                                                        size=(len(self.semantic_task[task]), 1))
        self.task_ebmedding = []
        self.task2id = {}
        for num, i in enumerate(list(self.semantic_task.keys())):
            self.task2id[i] = num
            self.task_ebmedding.append(self.semantic_task[i])
        self.task_ebmedding = torch.tensor(np.vstack(self.task_ebmedding)).float().to(self.device)

        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols

        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain,
                                     embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
                                     finetune=self.fine_tune, aggregate=self.aggregate,
                                     task_emb=self.task_ebmedding).to(self.device)
        self.matcher.eval()

        self.G_m = Generate_Model(in_dim=self.task_ebmedding.shape[1]).to(self.device)
        self.G_m.eval()

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

    def load_symbol2id(self): pass

    def load_embed(self):
        symbol_id = {}; rel2id = json.load(open(self.dataset+'/relation2ids'))
        ent2id = json.load(open(self.dataset+'/ent2ids'))
        rel2emb = json.load(open(self.dataset+'/relation2embids'))
        ent2emb = json.load(open(self.dataset+'/ent2embids'))
        ent_e = np.load(self.dataset+'/DRKG_TransE_entity.npy')
        rel_e = np.load(self.dataset+'/DRKG_TransE_relation.npy')
        i=0; emb=[]
        for k in rel2id:
            if k not in ['','OOV']:
                symbol_id[k]=i; i+=1
                emb.append(list(rel_e[rel2emb[k],:]) if rel2emb[k]!=-1 else list(np.random.randn(rel_e.shape[1])))
        for k in ent2id:
            if k not in ['','OOV']:
                symbol_id[k]=i; i+=1
                emb.append(list(ent_e[ent2emb[k],:]) if ent2emb[k]!=-1 else list(np.random.randn(rel_e.shape[1])))
        symbol_id['PAD']=i; emb.append(list(np.zeros((rel_e.shape[1],))))
        self.symbol2id=symbol_id; self.symbol2vec=np.array(emb)

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2))*self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list); self.e1_degrees = defaultdict(int)
        with open(self.dataset+'/path_graph') as f:
            for line in tqdm(f.readlines(), desc='Building connections'):
                e1,rel,e2 = line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel], self.symbol2id[e2]))
        for ent, id_ in self.ent2id.items():
            nb = self.e1_rele2[ent]
            if len(nb)>max_: random.shuffle(nb); nb=nb[:max_]
            self.e1_degrees[id_]=len(nb)
            for idx,_ in enumerate(nb):
                self.connections[id_,idx,0]=_[0]; self.connections[id_,idx,1]=_[1]

    def get_meta(self, left, right):
        lc = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).to(self.device)
        ld = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        rc = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).to(self.device)
        rd = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (lc, ld, rc, rd)

    def load_models(self, matcher_path, g_path):
        ms = torch.load(matcher_path, map_location=self.device)

        # ---- Convert 1-output fc to 2-output EDL head (WITH WARNING) ----
        convert_fc_1to2(ms)

        # 清理不兼容 keys
        for k in list(ms.keys()):
            if 'symbol_emb' in k or 'gcn_w' in k or 'gcn_b' in k or 'Bilinear' in k or \
               'Linear_self' in k or 'Linear_nei' in k or 'Linear_weak_rel' in k or \
               'NeighborAggregator' in k or 'siamese' in k or 'support_encoder' in k or \
               'query_encoder' in k:
                del ms[k]
        load_state_dict_safe(self.matcher, ms, model_name='matcher')
        logging.info(f'Matcher loaded from {matcher_path}')

        # 加载 G_m
        self.G_m = torch.load(g_path, map_location=self.device)
        self.G_m.eval()
        logging.info(f'Generator loaded from {g_path}')

        # ---- Patch forward: 返回 prob + alpha（修复 double-sigmoid bug）----
        def eviddie_forward(matcher_self, task_proto, query=None, support=None,
                            query_meta=None, support_meta=None,
                            query_batch=None, support_batch=None,
                            optim_VAE=None, is_eval=True, trainGAN=False):

            # CSE 编码
            query_left_, query_right_ = matcher_self.model(query_batch)
            query_neighbor = torch.cat((query_left_, query_right_), dim=-1)

            # VAE (eval mode: fixed epsilon for determinism)
            output_q, z_mean_q, z_logvar_q, zq = matcher_self.vaemodel(
                query_neighbor, is_support=False, is_eval=True)

            # EDL evidence — fc outputs 2-dim [evidence_0, evidence_1]
            evidence = F.softplus(matcher_self.fc(
                torch.abs(task_proto.expand_as(zq) - zq)))
            alpha = evidence + 1

            # Dirichlet expectation: p_1 = alpha_1 / (alpha_0 + alpha_1)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            # Epistemic uncertainty: u = K / sum(alpha), K=2
            u = 2.0 / torch.sum(alpha, dim=1)

            return prob[:, 1], u

        self.matcher.forward = types.MethodType(eviddie_forward, self.matcher)
        logging.info('Forward patched: returns (prob_class1, epistemic_uncertainty)')

    def export(self, mode, csv_writer, seed, method_name):
        symbol2id = self.symbol2id
        setting_map = {'dev':'common', 'test':'fewer', 'test2':'rare'}
        setting = setting_map.get(mode, mode)
        logging.info(f'EVALUATING {mode.upper()} (setting={setting})')

        if mode=='dev':
            test_tasks = json.load(open(self.dataset+'/dev_tasks.json'))
        elif mode=='test':
            test_tasks = json.load(open(self.dataset+'/test_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset+'/test2_tasks.json'))
        rel2id = json.load(open(self.dataset+'/relation2ids'))

        all_probs, all_gts = [], []
        with torch.no_grad():
            for query_ in test_tasks.keys():
                candidates = self.rel2candidates[query_]
                few = 0  # zero-shot
                query_triples = test_tasks[query_][few:]
                if not query_triples: continue

                false_triples = []
                np.random.seed(seed + sum(ord(c) for c in query_) % 10000)
                for t in query_triples:
                    e_h, rel, e_t = t[0], t[1], t[2]
                    while True:
                        noise = np.random.choice(candidates)
                        if (noise not in self.e1rel_e2.get(e_h+rel,[])) and noise!=e_t:
                            break
                    false_triples.append([e_h, rel, noise])

                all_triples = query_triples + false_triples
                all_rel2id = [[t[0], t[2], rel2id[t[1]]] for t in all_triples]
                q_left = [self.ent2id[t[0]] for t in all_triples]
                q_right = [self.ent2id[t[2]] for t in all_triples]
                q_meta = self.get_meta(q_left, q_right)
                n_pos = len(query_triples)

                qb = DrugDataset(all_rel2id)
                qbl = DrugDataLoader(qb, batch_size=len(all_rel2id), shuffle=False)
                qb_data = [t.to(self.device) for t in next(iter(qbl))]

                task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()

                probs, uncs = self.matcher(task_emb, None, None, q_meta, None, qb_data, None, None, is_eval=True)

                probs_np = probs.cpu().numpy()
                uncs_np = uncs.cpu().numpy()
                gt = np.concatenate([np.ones(n_pos), np.zeros(len(all_triples)-n_pos)])

                for idx, (t, p, u) in enumerate(zip(all_triples, probs_np, uncs_np)):
                    csv_writer.writerow([seed, setting, 0, method_name, query_,
                                         t[0], t[2], int(gt[idx]), 1 if p>=0.5 else 0,
                                         round(float(p),8), round(float(u),8)])

                all_probs.append(probs_np); all_gts.append(gt)

        if all_probs:
            ap, ag = np.concatenate(all_probs), np.concatenate(all_gts)
            acc, auc, f1 = do_compute_metrics(ap, ag)
            logging.info(f'  [{mode.upper()}] SUMMARY: acc={acc:.4f}, auc={auc:.4f}, f1={f1:.4f}')


if __name__ == '__main__':
    args = read_options()
    args.dataset = 'dataset1'
    args.no_meta = False

    SEEDS = [2024, 2025, 2026, 2027, 2028]
    MODES = ['dev', 'test', 'test2']
    METHOD = 'EviDDIE'

    MATCHER_PATH = 'models/dataset1/bestmodels'
    G_PATH = 'models/dataset1/bestmodels_G'

    out_dir = 'results/predictions'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'predictions_dataset1_EviDDIE.csv')

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed','setting','shot','method','event_type','drug_a','drug_b',
                     'y_true','y_pred','prob','uncertainty'])

        
        log_seed_checkpoint_note(MATCHER_PATH, SEEDS)
        for seed in SEEDS:
            logging.info(f'=== SEED {seed} ===')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            args.seed = seed

            ex = EviDDIEExport(args)
            ex.load_models(MATCHER_PATH, G_PATH)

            for mode in MODES:
                ex.export(mode, w, seed, METHOD)

    logging.info(f'Done! Saved to {out_csv}')
