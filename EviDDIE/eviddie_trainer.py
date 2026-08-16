#!/usr/bin/env Python
# coding=utf-8

import json
import logging
import sys
import os

# 允许从任意目录启动：把仓库根目录加入 sys.path（shared/ 位于仓库根）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from eviddie_args import read_options
from eviddie_dataloader import *
from eviddie_matcher import *
from tensorboardX import SummaryWriter

from tqdm import tqdm
import pickle
from torch_geometric.data import Batch, Data
from sklearn import metrics
from eviddie_recorder import ExperimentRecorder
from shared.eval_manifest import load_fixed_event_rows  # P0-4: fixed-manifest evaluation data builder


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):

    def __init__(self, arg):
        super(Trainer, self).__init__()
        self.device = DEVICE
        for k, v in vars(arg).items(): setattr(self, k, v)
        if not hasattr(self, "zero_shot"):
            self.zero_shot = False
        self.meta = not self.no_meta

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
        else:
            self.load_embed()
        self.use_pretrain = use_pretrain

        self.semantic_task = json.load(open(f'{self.dataset}/{self.semantic}'))

        # P0-7: training-time semantic augmentation only. Prototypes are
        # perturbed during training for robustness; all inference/export
        # paths use the raw BioSentVec embeddings without noise.
        # P0-2 audit: explicit 700-dim shape check + fixed key ordering + rng-seeded noise.
        rng = np.random.default_rng(self.seed)
        ordered_keys = sorted(list(self.semantic_task.keys()))  # fixed ordering to prevent prototype misalignment
        noise_scale = float(getattr(self, 'semantic_noise', 0.3))
        for task in tqdm(ordered_keys):
            vector = np.asarray(self.semantic_task[task], dtype=np.float32).reshape(-1)
            if vector.shape != (700,):
                raise ValueError(f'{task}: expected 700-d BioSentVec vector, got {vector.shape}')
            noise = rng.normal(loc=0.0, scale=noise_scale, size=vector.shape) if noise_scale > 0 else 0.0
            self.semantic_task[task] = vector + noise

        self.task_ebmedding = []
        self.task2id = {}
        for num, i in enumerate(ordered_keys):
            self.task2id[i] = num
            self.task_ebmedding.append(self.semantic_task[i])

        self.task_ebmedding = torch.tensor(np.vstack(self.task_ebmedding)).float().to(self.device)

        # P0-4: dev checkpoint selection uses the fixed manifest (hash recorded)
        self.dev_rows, self.dev_manifest_sha256 = load_fixed_event_rows(
            self.dataset, split='dev',
            manifest_seed=getattr(self, 'eval_manifest_seed', 19940419))
        logging.info(f'[P0-4] Fixed dev manifest loaded: {len(self.dev_rows)} rows, '
                     f'sha256={self.dev_manifest_sha256}')

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain,
                                    embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
                                    finetune=self.fine_tune, aggregate=self.aggregate, task_emb=self.task_ebmedding)
        self.matcher.to(self.device)

        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        self.optim_VAE = optim.Adam(self.matcher.vaemodel.parameters(), lr=self.lr * 10, weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        # KG 恢复 (2026-08-16)：get_meta/matcher forward 需要 DRKG 邻居连接与度数
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        self.all_drug_data = {}
        self.drug_num_node_indices = {}

        self.G_m = Generate_Model(in_dim=self.task_ebmedding.shape[1]).to(self.device)
        self.D_m = Distinguish_Model().to(self.device)
        self.D_optim = torch.optim.Adam(self.D_m.parameters(), lr=1e-4)

        self.G_optim = torch.optim.Adam(self.G_m.parameters(), lr=1e-4)

        # initialize the experiment recorder
        result_file = f'result_{self.prefix}.txt' if hasattr(self, 'prefix') else 'result.txt'
        self.recorder = ExperimentRecorder(project_name="ZetaDDIE", result_file=result_file)
        self.recorder.record_hyperparameters(arg)

    def load_symbol2id(self):
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        relation2embids = json.load(open(self.dataset + '/relation2embids'))
        ent2embids = json.load(open(self.dataset + '/ent2embids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.load(self.dataset + '/DRKG_' + self.embed_model + '_entity.npy')
            rel_embed = np.load(self.dataset + '/DRKG_' + self.embed_model + '_relation.npy')

            if self.embed_model == 'ComplEx':
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    if relation2embids[key] == -1:
                        embeddings.append(list(np.random.randn(rel_embed.shape[1], )))
                    else:
                        embeddings.append(list(rel_embed[relation2embids[key], :]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    if ent2embids[key] == -1:
                        embeddings.append(list(np.random.randn(rel_embed.shape[1], )))
                    else:
                        embeddings.append(list(ent_embed[ent2embids[key], :]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel], self.symbol2id[e2]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            # KG 恢复 (2026-08-16)：dataset1 有 8909/16837 个实体不在 path_graph；
            # 无邻居实体 -> 空列表 -> 度数 0、全 PAD，neighbor_encoder 靠 PAD 掩码 + 残差门控退化为自身特征
            neighbors = self.e1_rele2.get(ent, [])
            if len(neighbors) > max_:
                random.shuffle(neighbors)
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path)

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        # KG 恢复 (2026-08-16)：matcher forward 需要 DRKG 邻居 meta；
        # 实现与 eviddie_export_zs_v2.py 的 get_meta 保持一致（self.device 替代 .cuda()）。
        left_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).to(self.device)
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        right_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).to(self.device)
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        if self.zero_shot:
            self.train_zero_shot()
        else:
            self.train_standard()

    def train_standard(self):
        """Standard training loop."""
        logging.info('START STANDARD TRAINING...')
        losses = deque([], self.log_every)

        probas_pred = []
        ground_truth = []
        bestvalauc = 0
        bestvalap = 0
        lT = [torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), ]

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id,
                                   self.e1rel_e2, self.all_drug_data, self.drug_num_node_indices):

            if self.batch_nums % 50 == 0:
                logging.info('CURRENT EPOCH: %d MAX EPOCH %d' % (self.batch_nums, self.max_batches))
            task_name, support, query, false, support_left, support_right, query_left, query_right, false_left, false_right, support_batch, query_batch, false_batch = data
            support_batch = [t.to(self.device) for t in support_batch]
            query_batch = [t.to(self.device) for t in query_batch]
            false_batch = [t.to(self.device) for t in false_batch]
            # TODO more elegant solution
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).to(self.device)
            query = Variable(torch.LongTensor(query)).to(self.device)
            false = Variable(torch.LongTensor(false)).to(self.device)

            self.matcher.eval()
            zs = self.matcher(self.task_ebmedding[self.task2id[task_name]], query, support, query_meta,
                              support_meta, query_batch, support_batch, self.optim_VAE, trainGAN=True)
            dis_loss_all = 0
            gen_loss_all = 0


            zsi = zs.detach()
            Dis_true = self.D_m(zsi)
            true_loss = torch.nn.BCELoss()(Dis_true, torch.ones_like(Dis_true))
            fake_sample = self.G_m(self.task_ebmedding[self.task2id[task_name]])
            Dis_fake = self.D_m(fake_sample.detach())
            fake_loss = torch.nn.BCELoss()(Dis_fake, torch.zeros_like(Dis_fake))
            Dis_loss = true_loss + fake_loss
            self.D_optim.zero_grad()
            Dis_loss.backward()
            self.D_optim.step()
            Dis_G = self.D_m(fake_sample)
            G_loss = torch.nn.BCELoss()(Dis_G, torch.ones_like(Dis_G))
            self.G_optim.zero_grad()
            G_loss.backward()
            self.G_optim.step()
            with torch.no_grad():
                dis_loss_all += Dis_loss
                gen_loss_all += G_loss

            lT[3] = dis_loss_all
            lT[4] = gen_loss_all
            self.matcher.train()
            self.G_m.eval()
            task_emb = self.G_m(self.task_ebmedding[self.task2id[task_name]]).detach()
            self.G_m.train()
            ####old##############new#######################
            # # legacy: query_scores actually returned the alpha parameters
            # query_alpha, loss2_p = self.matcher(task_emb, query, support, query_meta, support_meta, query_batch,
            #                                     support_batch, self.optim_VAE, trainGAN=False)
            # false_alpha, loss2_n = self.matcher(task_emb, query, support, false_meta, support_meta, false_batch,
            #                                     support_batch, self.optim_VAE, trainGAN=False)
            #
            # # legacy: use the new evidential loss
            # loss_p = loss_fn(query_alpha, 'pos', self.batch_nums)
            # loss_n = loss_fn(false_alpha, 'neg', self.batch_nums)
            # loss_main = (loss_p + loss_n) / 2
            #
            # # legacy: compute predicted probabilities for logging
            # with torch.no_grad():
            #     q_p = query_alpha / torch.sum(query_alpha, dim=1, keepdim=True)
            #     f_p = false_alpha / torch.sum(false_alpha, dim=1, keepdim=True)
            #     probas_pred.append(np.concatenate([q_p[:, 1].cpu(), f_p[:, 1].cpu()]))
            #
            # loss = loss_main + loss2_p + loss2_n
            #
            #
            #
            #
            # lT[0] = loss.detach()
            # loss += loss2

            ####################new#############
            # 1. Forward pass: obtain evidence alpha and the SRAE loss
            query_alpha, loss2_p = self.matcher(task_emb, query, support, query_meta, support_meta, query_batch,
                                                support_batch, self.optim_VAE, trainGAN=False)
            false_alpha, loss2_n = self.matcher(task_emb, query, support, false_meta, support_meta, false_batch,
                                                support_batch, self.optim_VAE, trainGAN=False)

            # 2. Compute the evidential loss
            loss_p = loss_fn(query_alpha, 'pos', self.batch_nums)
            loss_n = loss_fn(false_alpha, 'neg', self.batch_nums)
            loss_main = (loss_p + loss_n) / 2

            # 3. Compute probabilities and labels for logging
            with torch.no_grad():
                q_p = query_alpha / torch.sum(query_alpha, dim=1, keepdim=True)
                f_p = false_alpha / torch.sum(false_alpha, dim=1, keepdim=True)
                probas_pred.append(np.concatenate([q_p[:, 1].cpu().numpy(), f_p[:, 1].cpu().numpy()]))
                # complete the ground_truth record, otherwise metric computation fails
                ground_truth.append(np.concatenate([np.ones(q_p.shape[0]), np.zeros(f_p.shape[0])]))

            # 4. Total loss: EDL + SRAE reconstruction only
            loss = (2.0 * loss_main) + loss2_p + loss2_n

            lT[0] = loss_main.detach()
            lT[1] = loss2_p.detach()
            lT[2] = loss2_n.detach()
            losses.append(loss.item())

            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (self.batch_nums + 1) % self.eval_every == 0:
                # P0-4: dev checkpoint evaluation uses the fixed manifest (no random.choice)
                dev_metrics = self.evaluate_fixed_dev()
                valauc = dev_metrics['pooled_auroc']
                valap = dev_metrics['pooled_auprc']

                is_best = (valauc > bestvalauc) or (np.isclose(valauc, bestvalauc) and valap > bestvalap)
                if is_best:
                    bestvalauc = valauc
                    bestvalap = valap
                    self.save(self.save_path + f'bestmodel')
                    torch.save(self.G_m, self.save_path + f'bestmodel_G')  # save generator
                    torch.save(self.D_m, self.save_path + f'bestmodel_D')  # save critic
                    # P0-3: save checkpoint metadata (dev manifest hash fixed under P0-4)
                    save_checkpoint_metadata(
                        self.save_path + 'bestmodel_meta.json',
                        train_seed=getattr(args, 'seed', None),
                        best_step=self.batch_nums,
                        dev_metric_name='dev_auroc',
                        dev_metric_value=float(valauc),
                        dev_manifest_sha256=self.dev_manifest_sha256,
                    )
                    # record the latest evaluation as best
                    if hasattr(self, 'recorder'):
                        self.recorder.experiment_data['best_models']['dev'] = {
                            'batch_num': self.batch_nums,
                            'metrics': {'auroc': valauc},
                            'timestamp': self.recorder.experiment_data['evaluation_results']['dev'][-1][
                                'timestamp'] if 'dev' in self.recorder.experiment_data['evaluation_results'] else ''
                        }
                        self.recorder._write_to_file()

            if self.batch_nums % self.log_every == 0:
                self.writer.add_scalar('Avg_batch_loss', np.mean(losses), self.batch_nums)
                acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred),
                                                                                         np.concatenate(ground_truth))
                logging.info(
                    f'loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')
                # log the training step
                metrics_dict = {
                    'acc': acc, 'auroc': auroc, 'f1_score': f1_score,
                    'precision': precision, 'recall': recall, 'int_ap': int_ap, 'ap': ap
                }
                self.recorder.record_training_step(self.batch_nums, loss.item(), metrics_dict)

            self.batch_nums += 1
            self.scheduler.step()
            if self.batch_nums == self.max_batches:
                self.save()
                self.recorder.finalize()
                break
    def _predict_fixed_rows(self, rows):
        """P0-4: forward fixed-manifest rows (positive/negative interleaved) per event; returns (probs, labels)."""
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        rows_by_event = {}
        for (evt, head, rel, tail, lab) in rows:
            rows_by_event.setdefault(evt, []).append((head, rel, tail, lab))
        probas, labels = [], []
        for evt, ev_rows in tqdm(rows_by_event.items(), desc='Fixed eval'):
            triples = [[h, t, rel2id[r]] for (h, r, t, _) in ev_rows]
            labels_e = [lab for (_, _, _, lab) in ev_rows]
            test_size = self.batch_size * 20
            for i in range(0, len(triples), test_size):
                batch_triples = triples[i:i + test_size]
                batch_labels = labels_e[i:i + test_size]
                batch_pairs = torch.LongTensor(
                    [[self.symbol2id[h], self.symbol2id[t]] for h, t, _ in batch_triples]
                ).to(self.device)
                ql = [self.ent2id[t[0]] for t in batch_triples]
                qr = [self.ent2id[t[1]] for t in batch_triples]
                query_meta = self.get_meta(ql, qr)
                batch_data = DrugDataset(batch_triples)
                loader = DrugDataLoader(batch_data, batch_size=len(batch_triples), shuffle=False)
                query_batch = [t.to(self.device) for t in next(iter(loader))]
                self.G_m.eval()
                task_emb = self.G_m(self.task_ebmedding[self.task2id[evt]]).detach()
                with torch.no_grad():
                    scores_prob, _ = self.matcher(task_emb, batch_pairs, None, query_meta, None,
                                                  query_batch, None, self.optim_VAE,
                                                  is_eval=True, trainGAN=False)
                probas.append(scores_prob.detach().cpu().numpy())
                labels.append(np.asarray(batch_labels))
        if not probas:
            return np.zeros(0), np.zeros(0, dtype=int)
        return np.concatenate(probas), np.concatenate(labels)

    def evaluate_fixed_dev(self):
        """P0-4: dev checkpoint selection evaluates on the fixed manifest (no random.choice)."""
        self.matcher.eval()
        yp, yt = self._predict_fixed_rows(self.dev_rows)
        if len(yp) == 0:
            logging.error('Fixed dev eval produced no predictions!')
            return {'pooled_auroc': 0.0, 'pooled_auprc': 0.0, 'acc': 0.0, 'f1': 0.0}
        acc, auroc, f1, pre, rec, int_ap, ap = do_compute_metrics(yp, yt)
        logging.info(f'[DEV-FIXED] ROC: {auroc:.4f} | AP: {ap:.4f} | ACC: {acc:.4f} | F1: {f1:.4f} '
                     f'(manifest sha256={self.dev_manifest_sha256[:12]}...)')
        self.recorder.record_evaluation('dev', {'acc': acc, 'auroc': auroc, 'f1_score': f1, 'ap': ap},
                                        is_best=False, batch_num=self.batch_nums)
        self.matcher.train()
        return {'pooled_auroc': auroc, 'pooled_auprc': ap, 'acc': acc, 'f1': f1}

    def eval_acc(self, mode='dev', meta=False):
        # P0-4: test/test2 evaluation is forbidden during training; allowed only after checkpoint lock (test_())
        if mode != 'dev' and not getattr(self, '_locked_eval', False):
            raise RuntimeError(
                'test/test2 evaluation is forbidden before checkpoint locking (P0-4). '
                'Use the locked evaluation entry (--test) after training.')
        # P0-4: all evaluation modes read the fixed manifest (no random.choice)
        split = {'dev': 'dev', 'test': 'test', 'test2': 'test2'}[mode]
        rows, mh = load_fixed_event_rows(self.dataset, split=split,
                                         manifest_seed=getattr(args, 'eval_manifest_seed', 19940419))
        logging.info('EVALUATING ON %s DATA (fixed manifest, sha256=%s...)' % (mode.upper(), mh[:12]))
        self.matcher.eval()
        yp, yt = self._predict_fixed_rows(rows)
        acc, auroc, f1, pre, rec, int_ap, ap = do_compute_metrics(yp, yt)
        logging.info(
            f'[{mode.upper()}] Global Metrics: ROC: {auroc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}')
        metrics_dict = {'acc': acc, 'auroc': auroc, 'f1_score': f1, 'ap': ap}
        self.recorder.record_evaluation(mode, metrics_dict, is_best=False, batch_num=self.batch_nums)
        self.matcher.train()
        return auroc
    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        # P0-4: test/test2 evaluation allowed only after checkpoint lock
        self._locked_eval = True
        testauc = self.eval_acc(meta=self.meta, mode='test')
        test2auc = self.eval_acc(meta=self.meta, mode='test2')
        # finalize test records
        self.recorder.finalize()


def make_target(target_type, batch_size, device='cpu'):
    """EDL training objective (class-order convention matches the export scripts;
    see tests/test_evidential_class_order.py).

    fc output channel convention: 0 = negative class, 1 = positive class.
    - positive 'pos' -> [0, 1]
    - negative 'neg' -> [1, 0]
    """
    if target_type == 'pos':
        y = torch.tensor([[0, 1]], dtype=torch.float32)
    elif target_type == 'neg':
        y = torch.tensor([[1, 0]], dtype=torch.float32)
    else:
        raise ValueError(f"target_type must be 'pos' or 'neg', got {target_type!r}")
    return y.repeat(batch_size, 1).to(device)


def save_checkpoint_metadata(path, train_seed, best_step, dev_metric_name,
                             dev_metric_value, dev_manifest_sha256):
    """P0-3: checkpoint metadata JSON (train seed / best step / class order / head type).

    Used by export/table scripts to verify that the matcher and generator of the
    same seed match and the class order is fixed.
    """
    payload = {
        'format_version': 2,
        'head_type': 'dirichlet_two_class',
        'class_order': ['negative', 'positive'],
        'train_seed': train_seed,
        'best_step': int(best_step),
        'dev_metric_name': dev_metric_name,
        'dev_metric_value': float(dev_metric_value),
        'dev_manifest_sha256': dev_manifest_sha256,
        'note': 'dev_manifest_sha256 filled in under P0-4 fixed-manifest dev selection',
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.info(f'Saved checkpoint metadata to {path}')


class EvidentialLoss(nn.Module):
    """
    EviDDIE evidential loss: expected mean-squared error plus an annealed
    KL regularizer over the Dirichlet evidence distribution.
    """

    def __init__(self, annealing_step=500):
        super(EvidentialLoss, self).__init__()
        self.annealing_step = annealing_step  # anneals the KL regularizer weight over training

    def kl_divergence(self, alpha):
        ones = torch.ones([1, 2], dtype=torch.float32).to(DEVICE)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + \
                     torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
        second_term = (alpha - ones).detach() * (torch.digamma(alpha) - torch.digamma(sum_alpha))
        return first_term + second_term.sum(dim=1, keepdim=True)

    def forward(self, alpha, target_type, batch_num):
        # target_type: 'pos' = positive, 'neg' = negative
        y = make_target(target_type, alpha.size(0), DEVICE)

        S = torch.sum(alpha, dim=1, keepdim=True)
        # 1. Expected mean squared error
        p = alpha / S
        err = (y - p) ** 2
        var = p * (1 - p) / (S + 1)
        loss_mse = torch.mean(torch.sum(err + var, dim=1))

        # 2. KL regularization (penalizes evidence of the incorrect class)
        annealing_coef = min(1.0, batch_num / self.annealing_step)
        alpha_hat = y + (1 - y) * alpha
        loss_kl = annealing_coef * torch.mean(self.kl_divergence(alpha_hat))

        return loss_mse + loss_kl

    ################################################


if __name__ == '__main__':
    args = read_options()
    # P0-7: per-seed independent checkpoints - the checkpoint prefix always
    # encodes the training seed, so the matcher (bestmodel) and generator
    # (bestmodel_G) checkpoints are never shared across seeds.
    args.save_path = f'models/{args.prefix}_seed{args.seed}'

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    device = DEVICE
    loss_fn = EvidentialLoss(annealing_step=10000)
    trainer = Trainer(args)
    if args.test:
        trainer.test_()
    else:
        trainer.train()