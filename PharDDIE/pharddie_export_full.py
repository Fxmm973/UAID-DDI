#!/usr/bin/env Python
# coding=utf-8
"""
导出完整 PharDDIE 模型（含 VAE/SRAE）的预测数据。
输出逐样本的概率和 VAE 潜在空间不确定性。
输出文件: results/predictions/predictions_dataset1_PharDDIE.csv
"""
import torch.nn as nn
import csv
from torch.autograd import Variable
from pharddie_args import read_options
from pharddie_dataloader import *
from pharddie_matcher import EmbedMatcher
from sklearn import metrics
from shared.checkpoint import load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred) if len(np.unique(target)) > 1 else 0.0
    f1 = metrics.f1_score(target, pred, zero_division=0)
    return acc, auroc, f1


class ExportFull(object):
    """导出完整 PharDDIE 模型（含 VAE）的逐样本预测和不确定性"""

    def __init__(self, arg):
        for k, v in vars(arg).items():
            setattr(self, k, v)
        self.meta = not self.no_meta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols

        # 完整 EmbedMatcher（含 model + neighbor_encoder + vaemodel + fc）
        self.matcher = EmbedMatcher(
            self.embed_dim, self.num_symbols, use_pretrain=True, embed=self.symbol2vec,
            dropout=self.dropout, batch_size=self.batch_size,
            finetune=self.fine_tune, aggregate=self.aggregate
        ).to(self.device)

        # 加载预训练权重
        ckpt = torch.load(arg.pretrained_model, map_location=self.device)
        for k in list(ckpt.keys()):
            if any(x in k for x in ['support_encoder.proj', 'support_encoder.layer_norm',
                                     'query_encoder.process', 'fc_struc_net']):
                del ckpt[k]
        load_state_dict_safe(self.matcher, ckpt, model_name='matcher')
        self.matcher.eval()
        for p in self.matcher.parameters():
            p.requires_grad = False

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
                embeddings.append(list(rel_embed[relation2embids[key], :]) if relation2embids[key] != -1
                                  else list(np.random.randn(rel_embed.shape[1])))
        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i; i += 1
                embeddings.append(list(ent_embed[ent2embids[key], :]) if ent2embids[key] != -1
                                  else list(np.random.randn(rel_embed.shape[1])))
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
        lc = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).to(self.device)
        ld = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        rc = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).to(self.device)
        rd = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (lc, ld, rc, rd)

    def encode_pairs_full(self, pairs_batch, left, right):
        """
        完整 PharDDIE 编码管线：
        MVN_DDI → neighbor_encoder → concat → VAE → (latent, uncertainty)
        返回: z_latent [B, 64], uncertainty [B]
        """
        meta = self.get_meta(left, right)
        lc, ld, rc, rd = meta

        # Step 1: 分子图编码 (MVN_DDI)
        ql_, qr_ = self.matcher.model(pairs_batch)

        # Step 2: 邻居编码 (ACI — bilinear attention over DRKG)
        ql = self.matcher.neighbor_encoder(lc, ld, ql_, qr_ - ql_)
        qr = self.matcher.neighbor_encoder(rc, rd, qr_, qr_ - ql_)

        # Step 3: 药物对表示
        pair_emb = torch.cat((ql, qr), dim=-1)  # [B, 256]

        # Step 4: VAE 编码 (SRAE) — 获取潜在码和不确定性
        y, z_mean, z_logvar, z = self.matcher.vaemodel(pair_emb, is_support=False, is_eval=True)

        # 不确定性: VAE 潜在空间逐样本平均方差
        # var = exp(logvar), 对所有潜在维度取均值
        uncertainty = torch.exp(z_logvar).mean(dim=-1)  # [B]

        return z, uncertainty  # [B, 64], [B]

    def export(self, mode, csv_writer, train_seed, eval_seed):
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

        all_p, all_u, all_gt = [], [], []
        with torch.no_grad():
            for query_ in test_tasks.keys():
                if len(test_tasks[query_]) < few + 1:
                    continue
                candidates = self.rel2candidates[query_]

                # ---- 支持集编码 ----
                support_triples = test_tasks[query_][:few]
                support_triples_rel2id = [[t[0], t[2], rel2id[t[1]]] for t in support_triples]
                support_left = [self.ent2id[t[0]] for t in support_triples]
                support_right = [self.ent2id[t[2]] for t in support_triples]

                sb = DrugDataset(support_triples_rel2id)
                sbl = DrugDataLoader(sb, batch_size=len(support_triples_rel2id), shuffle=False)
                sb_data = [t.to(self.device) for t in next(iter(sbl))]

                # 支持集也用完整管线编码，获取 latent code 计算原型
                s_meta = self.get_meta(support_left, support_right)
                slc, sld, src, srd = s_meta
                sl_, sr_ = self.matcher.model(sb_data)
                sl = self.matcher.neighbor_encoder(slc, sld, sl_, sr_ - sl_)
                sr = self.matcher.neighbor_encoder(src, srd, sr_, sr_ - sl_)
                s_pair = torch.cat((sl, sr), dim=-1)
                _, _, _, s_z = self.matcher.vaemodel(s_pair, is_support=True, is_eval=True)
                s_proto = s_z.mean(dim=0, keepdim=True)  # [1, 64]

                # ---- 查询集编码 ----
                query_triples = test_tasks[query_][few:]
                false_triples = []
                for t in query_triples:
                    while True:
                        noise = random.choice(candidates)
                        if (noise not in self.e1rel_e2[t[0] + t[1]]) and noise != t[2]:
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

                # 查询集完整管线编码
                zq, uncertainty = self.encode_pairs_full(qb_data, q_left, q_right)

                # ---- 评分 ----
                scores = self.matcher.fc(torch.abs(s_proto.expand_as(zq) - zq))
                probs = torch.sigmoid(scores).cpu().numpy().flatten()
                uncs = uncertainty.cpu().numpy().flatten()
                gt = np.concatenate([np.ones(n_pos), np.zeros(len(all_triples) - n_pos)])

                # ---- 写入 CSV ----
                for idx, (t, p, u) in enumerate(zip(all_triples, probs, uncs)):
                    csv_writer.writerow([
                        train_seed, eval_seed, setting, few, 'PharDDIE', query_,
                        t[0], t[2], int(gt[idx]), 1 if p >= 0.5 else 0,
                        round(float(p), 8), round(float(u), 8)
                    ])
                all_p.append(probs); all_u.append(uncs); all_gt.append(gt)

        if all_p:
            ap, ag = np.concatenate(all_p), np.concatenate(all_gt)
            acc, auc, f1 = do_compute_metrics(ap, ag)
            logging.info(f'  [{mode.upper()}] SUMMARY: acc={acc:.4f}, auc={auc:.4f}, f1={f1:.4f}')


if __name__ == '__main__':
    args = read_options()
    # ---- 训练种子（独立训练运行）vs 负样本种子（固定评估） ----
    # 训练种子：每次从头训练产生不同模型 -> 捕捉训练不稳定性
    # 负样本种子：固定为 19940419，使跨训练种子的比较不受负样本波动干扰
    TRAINING_SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
    EVAL_MANIFEST_SEED = 19940419  # 固定负样本种子用于评估
    SHOTS = [1, 5, 10]
    MODES = ['dev', 'test', 'test2']
    DATASET = 'dataset1'

    output_dir = 'results/predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'predictions_dataset1_PharDDIE.csv')

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['train_seed', 'eval_seed', 'setting', 'shot', 'method', 'event_type',
                     'drug_a', 'drug_b', 'y_true', 'y_pred', 'prob', 'uncertainty'])

        logging.info('Exporting full PharDDIE model (with VAE uncertainty)')
        logging.info(f'Training seeds: {TRAINING_SEEDS}')
        logging.info(f'Fixed evaluation manifest seed: {EVAL_MANIFEST_SEED}')
        logging.info(f'Shots: {SHOTS}, Modes: {MODES}')

        for train_seed in TRAINING_SEEDS:
            logging.info(f'=== TRAIN_SEED {train_seed} ===')
            for few in SHOTS:
                args.few = few; args.train_few = few; args.dataset = DATASET
                # 每个训练种子对应独立 checkpoint
                args.pretrained_model = f'models/dataset1/models_drugbank_{few}shot_str_seed{train_seed}/bestmodel'
                # 回退：如果 per-seed checkpoint 不存在，尝试原始路径（向后兼容）
                if not os.path.exists(args.pretrained_model):
                    fallback = f'models/dataset1/models_drugbank_{few}shot_str/bestmodel'
                    if os.path.exists(fallback):
                        logging.warning(f'Per-seed checkpoint not found, using fallback: {fallback}')
                        logging.warning(f'  SD will reflect negative-sampling variation, not training-seed variation')
                        args.pretrained_model = fallback
                    else:
                        logging.warning(f'Checkpoint not found: {args.pretrained_model}, skipping')
                        continue

                # 使用固定负样本种子进行确定性评估
                eval_seed = EVAL_MANIFEST_SEED
                random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed)

                ex = ExportFull(args)
                for mode in MODES:
                    ex.export(mode, w, train_seed, eval_seed)

    logging.info(f'Done! Saved to {output_csv}')
    logging.info('NOTE: train_seed column identifies independent training runs.')
    logging.info('      eval_seed column identifies the fixed negative-sampling manifest used.')
    logging.info('      Mean +/- std across train_seeds = training variability.')
    logging.info('      Mean +/- std across eval_seeds = negative-sampling variability.')
