#!/usr/bin/env Python
# coding=utf-8
import torch.nn as nn
import csv
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

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

        ckpt = torch.load(arg.pretrained_model, map_location=self.device)
        for k in list(ckpt.keys()):
            if any(x in k for x in ['support_encoder.proj', 'support_encoder.layer_norm',
                                     'query_encoder.process', 'fc_struc_net']):
                del ckpt[k]
        load_state_dict_safe(self.matcher, ckpt, model_name='matcher')
        self.matcher.eval()
        for p in self.matcher.parameters():
            p.requires_grad = False

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
        with open(self.dataset + '/path_graph_train_only') as f:
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
                q_meta = self.get_meta(q_left, q_right)
                q_emb = self.get_neighbor_emb(qb_data, q_meta)

                scores = self.fc_direct(torch.abs(s_mean.expand_as(q_emb) - q_emb))
                probs = torch.sigmoid(scores).cpu().numpy().flatten()
                gt = np.concatenate([np.ones(n_pos), np.zeros(len(all_triples)-n_pos)])

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
    import hashlib
    SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
    SHOTS = [1, 5]  # 论文仅报告 1/5-shot；10-shot 模型不再要求
    MODES = ['dev', 'test', 'test2']
    DATASET = 'dataset1'

    hash_log = json.load(open(f'{DATASET}/neg_manifests/manifest_hashes.json'))
    for split in MODES:
        for seed in SEEDS:
            mf = f'{DATASET}/neg_manifests/{split}_seed{seed}_negatives.json'
            actual = hashlib.sha256(open(mf, 'rb').read()).hexdigest()
            recorded = hash_log.get(f'{split}_seed{seed}', {}).get('sha256')
            if recorded is None or actual != recorded:
                raise RuntimeError(f'Manifest hash mismatch: {mf} (recorded={recorded}, actual={actual})')
    logging.info('[MANIFEST-CHAIN] All evaluation manifest SHA256 verified.')

    output_dir = 'results/predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'predictions_dataset1_wo_uncertainty.csv')

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'setting', 'shot', 'method', 'event_type',
                     'drug_a', 'drug_b', 'y_true', 'y_pred', 'prob', 'uncertainty'])

        logging.info('Evaluating w/o-uncertainty variant on the fixed (non-seed) checkpoint '
                     'with the five fixed negative manifests.')
        for seed in SEEDS:
            logging.info(f'=== EVAL SEED {seed} ===')
            for few in SHOTS:
                args.few = few; args.train_few = few; args.dataset = DATASET
                args.eval_seed = seed
                args.pretrained_model = f'models/dataset1/models_drugbank_{few}shot_str/bestmodel'
                args.fc_direct_path = f'models/dataset1/models_wo_uncertainty_{few}shot/bestmodel'
                if not os.path.exists(args.fc_direct_path):
                    raise FileNotFoundError(
                        f'fc_direct model not found: {args.fc_direct_path}. '
                        f'Refusing to skip: the w/o-uncertainty export must cover all shots.')
                if not os.path.exists(args.pretrained_model):
                    raise FileNotFoundError(
                        f'Base checkpoint not found: {args.pretrained_model}.')


                ex = ExportWOUncertainty(args)
                for mode in MODES:
                    ex.export(mode, w, seed)

    # 完成后校验：CSV 必须包含数据行（5 eval seed x 2 shot x 3 分片组合都应产出），
    # 空 CSV 视为失败，禁止以退出码 0 结束。
    with open(output_csv, 'r', encoding='utf-8') as chk:
        n_rows = sum(1 for _ in chk) - 1
    if n_rows <= 0:
        logging.error(f'Export produced an empty CSV ({output_csv}); failing the step.')
        raise SystemExit(1)
    logging.info(f'Validation: {n_rows} data rows written to {output_csv}.')
    logging.info(f'Done! Saved to {output_csv}')
