#!/usr/bin/env Python
# coding=utf-8
"""
导出 w/o uncertainty 变体的预测数据。
使用冻结的编码器 + 新训练的 fc_direct 头（无 VAE）。
"""
import torch.nn as nn
import csv
from torch.autograd import Variable
from pharddie_args import read_options
from pharddie_dataloader import *
from pharddie_matcher import EmbedMatcher
from sklearn import metrics
from shared.checkpoint import load_state_dict_safe, log_seed_checkpoint_note

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred) if len(np.unique(target)) > 1 else 0.0
    f1 = metrics.f1_score(target, pred, zero_division=0)
    return acc, auroc, f1

class ExportWOUncertainty(object):
    def __init__(self, arg):
        for k, v in vars(arg).items():
            setattr(self, k, v)
        self.meta = not self.no_meta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols

        self.matcher = EmbedMatcher(
            self.embed_dim, self.num_symbols, use_pretrain=True, embed=self.symbol2vec,
            dropout=self.dropout, batch_size=self.batch_size,
            finetune=self.fine_tune, aggregate=self.aggregate
        ).to(self.device)

        # 加载预训练权重（修复维度不匹配）
        ckpt = torch.load(arg.pretrained_model, map_location=self.device)
        for k in list(ckpt.keys()):
            if any(x in k for x in ['support_encoder.proj', 'support_encoder.layer_norm',
                                     'query_encoder.process', 'fc_struc_net']):
                del ckpt[k]
        load_state_dict_safe(self.matcher, ckpt, model_name='matcher')
        self.matcher.eval()
        for p in self.matcher.parameters():
            p.requires_grad = False

        # 加载 fc_direct 头
        neighbor_dim = self.embed_dim * 2
        self.fc_direct = nn.Sequential(
            nn.Linear(neighbor_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        ).to(self.device)
        self.fc_direct.load_state_dict(torch.load(arg.fc_direct_path, map_location=self.device))
        self.fc_direct.eval()
        logging.info(f'Loaded fc_direct from {arg.fc_direct_path}')

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))
        self.all_drug_data = {}
        self.drug_num_node_indices = {}

    def load_embed(self):
        symbol_id = {}; rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        relation2embids = json.load(open(self.dataset + '/relation2embids'))
        ent2embids = json.load(open(self.dataset + '/ent2embids'))
        ent_embed = np.load(self.dataset + '/DRKG_TransE_entity.npy')
        rel_embed = np.load(self.dataset + '/DRKG_TransE_relation.npy')
        i = 0; embeddings = []
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i; i += 1
                embeddings.append(list(rel_embed[relation2embids[key],:]) if relation2embids[key]!=-1 else list(np.random.randn(rel_embed.shape[1])))
        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i; i += 1
                embeddings.append(list(ent_embed[ent2embids[key],:]) if ent2embids[key]!=-1 else list(np.random.randn(rel_embed.shape[1])))
        symbol_id['PAD'] = i; embeddings.append(list(np.zeros((rel_embed.shape[1],))))
        self.symbol2id = symbol_id; self.symbol2vec = np.array(embeddings)

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list); self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            for line in tqdm(f.readlines()):
                e1, rel, e2 = line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel], self.symbol2id[e2]))
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                random.shuffle(neighbors); neighbors = neighbors[:max_]
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]; self.connections[id_, idx, 1] = _[1]

    def get_meta(self, left, right):
        lc = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).to(self.device)
        ld = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        rc = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).to(self.device)
        rd = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (lc, ld, rc, rd)

    def get_neighbor_emb(self, pairs_batch, meta):
        lc, ld, rc, rd = meta
        ql_, qr_ = self.matcher.model(pairs_batch)
        ql = self.matcher.neighbor_encoder(lc, ld, ql_, qr_ - ql_)
        qr = self.matcher.neighbor_encoder(rc, rd, qr_, qr_ - ql_)
        return torch.cat((ql, qr), dim=-1)

    def export(self, mode, csv_writer, seed):
        symbol2id = self.symbol2id; few = self.few
        setting_map = {'dev': 'common', 'test': 'fewer', 'test2': 'rare'}
        setting = setting_map.get(mode, mode)
        logging.info(f'EVALUATING {mode.upper()} (setting={setting})')

        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        elif mode == 'test':
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test2_tasks.json'))
        rel2id = json.load(open(self.dataset + '/relation2ids'))

        all_p, all_gt = [], []
        with torch.no_grad():
            for query_ in test_tasks.keys():
                if len(test_tasks[query_]) < few + 1:
                    continue
                candidates = self.rel2candidates[query_]
                support_triples = test_tasks[query_][:few]
                support_triples_rel2id = [[t[0], t[2], rel2id[t[1]]] for t in support_triples]
                support_pairs = [[symbol2id[t[0]], symbol2id[t[2]]] for t in support_triples]
                support_left = [self.ent2id[t[0]] for t in support_triples]
                support_right = [self.ent2id[t[2]] for t in support_triples]

                sb = DrugDataset(support_triples_rel2id)
                sbl = DrugDataLoader(sb, batch_size=len(support_triples_rel2id), shuffle=False)
                sb_data = [t.to(self.device) for t in next(iter(sbl))]
                s_meta = self.get_meta(support_left, support_right)
                s_emb = self.get_neighbor_emb(sb_data, s_meta)
                s_mean = s_emb.mean(dim=0, keepdim=True)

                query_triples = test_tasks[query_][few:]
                false_triples = []
                for t in query_triples:
                    while True:
                        noise = random.choice(candidates)
                        if (noise not in self.e1rel_e2[t[0]+t[1]]) and noise != t[2]:
                            break
                    false_triples.append([t[0], t[1], noise])

                all_triples = query_triples + false_triples
                all_rel2id = [[t[0], t[2], rel2id[t[1]]] for t in all_triples]
                q_left = [self.ent2id[t[0]] for t in all_triples]
                q_right = [self.ent2id[t[2]] for t in all_triples]
                n_pos = len(query_triples)

                qb = DrugDataset(all_rel2id)
                qbl = DrugDataLoader(qb, batch_size=len(all_rel2id), shuffle=False)
                qb_data = [t.to(self.device) for t in next(iter(qbl))]
                q_meta = self.get_meta(q_left, q_right)
                q_emb = self.get_neighbor_emb(qb_data, q_meta)

                scores = self.fc_direct(torch.abs(s_mean.expand_as(q_emb) - q_emb))
                probs = torch.sigmoid(scores).cpu().numpy().flatten()
                gt = np.concatenate([np.ones(n_pos), np.zeros(len(all_triples)-n_pos)])

                # CSV
                for idx, (t, p) in enumerate(zip(all_triples, probs)):
                    csv_writer.writerow([
                        seed, setting, few, 'w/o uncertainty', query_,
                        t[0], t[2], int(gt[idx]), 1 if p >= 0.5 else 0,
                        round(float(p), 8), round(float(1-p if p>0.5 else p), 8)
                    ])
                all_p.append(probs); all_gt.append(gt)

        if all_p:
            ap, ag = np.concatenate(all_p), np.concatenate(all_gt)
            acc, auc, f1 = do_compute_metrics(ap, ag)
            logging.info(f'  [{mode.upper()}] SUMMARY: acc={acc:.4f}, auc={auc:.4f}, f1={f1:.4f}')

if __name__ == '__main__':
    args = read_options()
    SEEDS = [2024, 2025, 2026, 2027, 2028]
    SHOTS = [1, 5, 10]
    MODES = ['dev', 'test', 'test2']
    DATASET = 'dataset1'

    output_dir = 'results/predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'predictions_dataset1_wo_uncertainty.csv')

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'setting', 'shot', 'method', 'event_type',
                     'drug_a', 'drug_b', 'y_true', 'y_pred', 'prob', 'uncertainty'])

        
        log_seed_checkpoint_note(MATCHER_PATH, SEEDS)
        for seed in SEEDS:
            logging.info(f'=== SEED {seed} ===')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            for few in SHOTS:
                args.few = few; args.train_few = few; args.dataset = DATASET
                args.pretrained_model = f'models/dataset1/models_drugbank_{few}shot_str/bestmodel'
                args.fc_direct_path = f'models/dataset1/models_wo_uncertainty_{few}shot/bestmodel'
                if not os.path.exists(args.fc_direct_path):
                    logging.warning(f'fc_direct model not found: {args.fc_direct_path}, skipping')
                    continue
                ex = ExportWOUncertainty(args)
                for mode in MODES:
                    ex.export(mode, w, seed)

    logging.info(f'Done! Saved to {output_csv}')
