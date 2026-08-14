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
        # ---- 固定负样本 manifest（正式评估路径：禁止现场抽样）----
        self.neg_manifests = {}
        for split in ['dev', 'test', 'test2']:
            mp = f'{self.dataset}/neg_manifests/{split}_seed{self.eval_seed}_negatives.json'
            if not os.path.exists(mp):
                raise FileNotFoundError(f'Evaluation manifest not found: {mp}. '
                                        'Run shared/neg_manifest.py first.')
            self.neg_manifests[split] = json.load(open(mp))
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

                # ---- 查询集编码：负样本直接读取固定 manifest（按 event/正样本/索引精确匹配）----
                query_triples = test_tasks[query_][few:]
                manifest_entries = self.neg_manifests[mode].get(query_, [])
                expected = manifest_entries[few:]
                if len(expected) != len(query_triples):
                    raise RuntimeError(
                        f'[{mode}] {query_}: manifest has {len(expected)} query negatives '
                        f'but task has {len(query_triples)} query positives.')
                false_triples = []
                for t, entry in zip(query_triples, expected):
                    d_i, d_j, d_k, rel = entry
                    if not (d_i == t[0] and d_j == t[2] and rel == t[1]):
                        raise RuntimeError(
                            f'[{mode}] {query_}: manifest entry {entry} does not match '
                            f'positive query triple {t} at the same index.')
                    false_triples.append([t[0], t[1], d_k])

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

    # ---- 运行前核验评估 manifest 的 SHA256（与 manifest_hashes.json 比对，不一致立即终止）----
    hash_log = json.load(open(f'{DATASET}/neg_manifests/manifest_hashes.json'))
    for split in MODES:
        mf = f'{DATASET}/neg_manifests/{split}_seed{EVAL_MANIFEST_SEED}_negatives.json'
        actual = hashlib.sha256(open(mf, 'rb').read()).hexdigest()
        recorded = hash_log.get(f'{split}_seed{EVAL_MANIFEST_SEED}', {}).get('sha256')
        if recorded is None or actual != recorded:
            raise RuntimeError(f'Manifest hash mismatch: {mf} (recorded={recorded}, actual={actual})')
        logging.info(f'[MANIFEST-CHAIN] {mf}: SHA256 verified.')

    output_dir = 'results/predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'predictions_dataset1_PharDDIE.csv')

    import hashlib  # 种子独立性验证
    ckpt_records = {}  # (shot, train_seed) -> checkpoint sha256

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
                # 每个训练种子必须使用对应种子训练出的独立 checkpoint；
                # 缺失时直接报错，绝不回退到其他检查点（避免把同一模型的重复推理误记为不同种子）
                args.pretrained_model = f'models/dataset1/models_drugbank_{few}shot_str_seed{train_seed}/bestmodel'
                if not os.path.exists(args.pretrained_model):
                    raise FileNotFoundError(
                        f'Per-seed checkpoint not found: {args.pretrained_model}. '
                        f'Per-seed exports must use the checkpoint of the corresponding training seed.')
                ckpt_records[(few, train_seed)] = hashlib.sha256(
                    open(args.pretrained_model, 'rb').read()).hexdigest()

                # 使用固定负样本种子进行确定性评估
                eval_seed = EVAL_MANIFEST_SEED
                random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed)

                args.eval_seed = eval_seed
                ex = ExportFull(args)
                for mode in MODES:
                    ex.export(mode, w, train_seed, eval_seed)

        # ---- 种子独立性验证：5 train_seed -> 5 不同 checkpoint 路径 -> 5 不同哈希 ----
        for few in SHOTS:
            hashes = [ckpt_records[(few, s)] for s in TRAINING_SEEDS]
            if len(set(hashes)) != len(hashes):
                raise RuntimeError(
                    f'{few}-shot: checkpoint hashes are not unique across training seeds: '
                    f'{len(set(hashes))} distinct of {len(hashes)}')
            logging.info(f'[SEED-CHAIN] {few}-shot: {len(set(hashes))} distinct checkpoint hashes '
                         f'across {len(hashes)} training seeds (OK).')
        logging.info(f'[SEED-CHAIN] Evaluation manifest seed fixed to {EVAL_MANIFEST_SEED} '
                     f'for all seeds (identical manifest hash across seeds).')

    logging.info(f'Done! Saved to {output_csv}')
    logging.info('NOTE: train_seed column identifies independent training runs.')
    logging.info('      eval_seed column identifies the fixed negative-sampling manifest used.')
    logging.info('      Mean +/- std across train_seeds = training variability.')
    logging.info('      Mean +/- std across eval_seeds = negative-sampling variability.')
