#!/usr/bin/env Python
# coding=utf-8

import json
import logging
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
        for k, v in vars(arg).items(): setattr(self, k, v)
        # =====新增 zero-shot flag (safe default) =====
        if not hasattr(self, "zero_shot"):
            self.zero_shot = False

            ###########新增结束
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

        self.semantic_task = json.load(open(f'{args.dataset}/{args.semantic}'))

        for task in tqdm(list(self.semantic_task.keys())):
            self.semantic_task[task] = np.array(self.semantic_task[task]) + 0.3 * np.random.normal(loc=0, scale=1,
                                                                                                   size=(len(
                                                                                                       self.semantic_task[
                                                                                                           task]), 1))

        self.task_ebmedding = []
        self.task2id = {}
        for num, i in enumerate(list(self.semantic_task.keys())):
            self.task2id[i] = num
            self.task_ebmedding.append(self.semantic_task[i])

        self.task_ebmedding = torch.tensor(np.vstack(self.task_ebmedding)).float().to(DEVICE)

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain,
                                    embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
                                    finetune=self.fine_tune, aggregate=self.aggregate, task_emb=self.task_ebmedding)
        self.matcher.to(DEVICE)

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

        # logging.info('BUILDING CONNECTION MATRIX')
        # degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        self.all_drug_data = {}
        self.drug_num_node_indices = {}

        self.G_m = Generate_Model(in_dim=self.task_ebmedding.shape[1]).to(DEVICE)
        self.D_m = Distinguish_Model().to(DEVICE)
        self.D_optim = torch.optim.Adam(self.D_m.parameters(), lr=1e-4)

        # ######################新增#######################################
        # # 初始化BioSentVec模型（条件导入，避免非零样本模式下的依赖）
        # self.biosent_model = None
        # if self.zero_shot:
        #     try:
        #         import gensim
        #         logging.info(f'Loading BioSentVec model from {self.biosent_path}')
        #         self.biosent_model = gensim.models.KeyedVectors.load_word2vec_format(
        #             self.biosent_path, binary=True
        #         )
        #         logging.info('BioSentVec model loaded successfully')
        #     except ImportError:
        #         logging.warning('Gensim not installed. Zero-shot learning will use random embeddings.')
        #         logging.warning('Install gensim with: pip install gensim')
        #     except Exception as e:
        #         logging.warning(f'Failed to load BioSentVec model: {e}')
        #         logging.warning('Zero-shot learning will use random embeddings.')
        #
        # # 为Mapper单独创建优化器（仅在零样本模式下使用）
        # if self.zero_shot:
        #     self.mapper_optim = torch.optim.Adam(
        #         self.matcher.enhanced_mapper.parameters(), lr=1e-4
        #     )
        # ######################新增结束#######################################

        self.G_optim = torch.optim.Adam(self.G_m.parameters(), lr=1e-4)

        # 初始化实验记录器
        result_file = f'result_{self.prefix}.txt' if hasattr(self, 'prefix') else 'result.txt'
        self.recorder = ExperimentRecorder(project_name="ZetaDDIE", result_file=result_file)
        self.recorder.record_hyperparameters(arg)

    # #################### # 新增为Mapper单独创建优化器（仅在零样本模式下使用）
    #      if hasattr(args, 'zero_shot') and args.zero_shot:
    #          self.mapper_optim = torch.optim.Adam(
    #              self.matcher.enhanced_mapper.parameters(), lr=1e-4
    #          )
    #
    #      ############新增结束####################3

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
            neighbors = self.e1_rele2[ent]
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
        return None
        # left_connections = Variable(
        #     torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
        # left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        # right_connections = Variable(
        #     torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
        # right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        # return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        if self.zero_shot:
            self.train_zero_shot()
        else:
            self.train_standard()

    def train_standard(self):
        """标准训练流程"""
        logging.info('START STANDARD TRAINING...')
        losses = deque([], self.log_every)

        probas_pred = []
        ground_truth = []
        bestvalauc = 0
        lT = [torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), ]

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id,
                                   self.e1rel_e2, self.all_drug_data, self.drug_num_node_indices):

            if self.batch_nums % 50 == 0:
                logging.info('CURRENT EPOCH: %d MAX EPOCH %d' % (self.batch_nums, self.max_batches))
            task_name, support, query, false, support_left, support_right, query_left, query_right, false_left, false_right, support_batch, query_batch, false_batch = data
            support_batch = [t.to(device) for t in support_batch]
            query_batch = [t.to(device) for t in query_batch]
            false_batch = [t.to(device) for t in false_batch]
            # TODO more elegant solution
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).to(DEVICE)
            query = Variable(torch.LongTensor(query)).to(DEVICE)
            false = Variable(torch.LongTensor(false)).to(DEVICE)

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
            # # 此时 query_scores 实际上返回的是 alpha 参数
            # query_alpha, loss2_p = self.matcher(task_emb, query, support, query_meta, support_meta, query_batch,
            #                                     support_batch, self.optim_VAE, trainGAN=False)
            # false_alpha, loss2_n = self.matcher(task_emb, query, support, false_meta, support_meta, false_batch,
            #                                     support_batch, self.optim_VAE, trainGAN=False)
            #
            # # 使用新的证据损失函数
            # loss_p = loss_fn(query_alpha, 'pos', self.batch_nums)
            # loss_n = loss_fn(false_alpha, 'neg', self.batch_nums)
            # loss_main = (loss_p + loss_n) / 2
            #
            # # 计算用于日志统计的预测概率
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
            # 1. 调用模型获取证据 alpha 和 VAE 损失
            query_alpha, loss2_p = self.matcher(task_emb, query, support, query_meta, support_meta, query_batch,
                                                support_batch, self.optim_VAE, trainGAN=False)
            false_alpha, loss2_n = self.matcher(task_emb, query, support, false_meta, support_meta, false_batch,
                                                support_batch, self.optim_VAE, trainGAN=False)

            # 2. 计算证据损失 (Evidential Loss)
            loss_p = loss_fn(query_alpha, 'pos', self.batch_nums)
            loss_n = loss_fn(false_alpha, 'neg', self.batch_nums)
            loss_main = (loss_p + loss_n) / 2

            # 3. 计算日志统计用的概率和标签
            with torch.no_grad():
                q_p = query_alpha / torch.sum(query_alpha, dim=1, keepdim=True)
                f_p = false_alpha / torch.sum(false_alpha, dim=1, keepdim=True)
                probas_pred.append(np.concatenate([q_p[:, 1].cpu().numpy(), f_p[:, 1].cpu().numpy()]))
                # 补全 ground_truth 记录，否则度量计算会报错
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
                valauc = self.eval_acc(meta=self.meta)

                is_best = valauc > bestvalauc
                if is_best:
                    bestvalauc = valauc
                    self.save(self.save_path + f'bestmodel')
                    torch.save(self.G_m, self.save_path + f'bestmodel_G')  # 保存模型
                    torch.save(self.D_m, self.save_path + f'bestmodel_D')  # 保存模型
                    # 更新最新评估为最佳
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
                # 记录训练步骤
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

    #
    # # ###########################新增#################################
    # def train_zero_shot(self):
    #     """零样本学习的训练流程 - 优化版"""
    #     logging.info('START ZERO-SHOT TRAINING WITH STRUCTURE GUIDANCE...')
    #
    #     if self.biosent_model is None:
    #         logging.error('BioSentVec model not loaded.')
    #         return
    #
    #     # 预计算所有事件的语义向量
    #     event_semantic_vecs = {}
    #     for event_name in self.semantic_task.keys():
    #         event_desc = event_name.replace('#Drug1', 'drug').replace('#Drug2', 'drug')
    #         words = event_desc.lower().split()
    #         word_vectors = [self.biosent_model[w] for w in words if w in self.biosent_model]
    #
    #         if word_vectors:
    #             vec = np.mean(word_vectors, axis=0)
    #         else:
    #             vec = np.random.randn(700)
    #
    #         event_semantic_vecs[event_name] = torch.FloatTensor(vec).unsqueeze(0).cuda()
    #
    #     best_val_auc = 0
    #
    #     for epoch in range(self.max_batches // 100):  # 调整epoch数
    #
    #         for data in train_generate(self.dataset, self.batch_size, self.train_few,
    #                                    self.symbol2id, self.ent2id, self.e1rel_e2,
    #                                    self.all_drug_data, self.drug_num_node_indices):
    #
    #             task_name, support, query, false, support_left, support_right, \
    #                 query_left, query_right, false_left, false_right, \
    #                 support_batch, query_batch, false_batch = data
    #
    #             # 转移到GPU
    #             support_batch = [t.to(device) for t in support_batch]
    #             query_batch = [t.to(device) for t in query_batch]
    #             false_batch = [t.to(device) for t in false_batch]
    #
    #             # 获取元数据
    #             support_meta = self.get_meta(support_left, support_right)
    #             query_meta = self.get_meta(query_left, query_right)
    #             false_meta = self.get_meta(false_left, false_right)
    #
    #             # ========== 阶段1：提取支持集的结构上下文 ==========
    #             self.matcher.eval()
    #             with torch.no_grad():
    #                 support_left_, support_right_ = self.matcher.model(support_batch)
    #                 support_left_feat = self.matcher.neighbor_encoder(
    #                     *support_meta[:2], support_left_, support_right_ - support_left_
    #                 )
    #                 support_right_feat = self.matcher.neighbor_encoder(
    #                     *support_meta[2:], support_right_, support_right_ - support_left_
    #                 )
    #                 support_neighbor = torch.cat([support_left_feat, support_right_feat], dim=-1)
    #                 _, _, _, support_context = self.matcher.vaemodel(
    #                     support_neighbor, is_support=True, is_eval=False
    #                 )
    #
    #             # ========== 阶段2：语义映射训练 ==========
    #             self.matcher.train()
    #             semantic_vec = event_semantic_vecs[task_name]
    #
    #             # 正样本预测（带结构引导）
    #             query_scores = self.matcher.forward_zero_shot(
    #                 semantic_vec, query_meta, query_batch, support_context
    #             )
    #
    #             # 负样本预测
    #             false_scores = self.matcher.forward_zero_shot(
    #                 semantic_vec, false_meta, false_batch, support_context
    #             )
    #
    #             # ========== 损失计算 ==========
    #             # 1. 主分类损失
    #             loss_main, _, _ = loss_fn(query_scores, false_scores)
    #
    #             # 2. 软对齐损失（拉近映射向量与支持集中心）
    #             mapped_vec = self.matcher.enhanced_mapper(semantic_vec, support_context)
    #             support_center = support_context.mean(dim=0, keepdim=True)
    #             loss_soft_align = F.mse_loss(mapped_vec, support_center)
    #
    #             # 3. 温度正则化（防止temperature过小）
    #             loss_temp_reg = torch.relu(0.01 - self.matcher.temperature)
    #
    #             # 4. 对比学习损失（增强判别性）
    #             # 让同一事件的样本在映射空间中更接近
    #             pos_sim = F.cosine_similarity(mapped_vec, support_center, dim=-1)
    #             loss_contrastive = -torch.log(torch.sigmoid(pos_sim / self.matcher.temperature))
    #
    #             # 总损失
    #             total_loss = (loss_main +
    #                           0.3 * loss_soft_align +
    #                           0.1 * loss_temp_reg +
    #                           0.2 * loss_contrastive)
    #
    #             # 反向传播
    #             self.optim.zero_grad()
    #             self.mapper_optim.zero_grad()
    #             total_loss.backward()
    #
    #             # 梯度裁剪
    #             torch.nn.utils.clip_grad_norm_(self.matcher.parameters(), 1.0)
    #
    #             self.optim.step()
    #             self.mapper_optim.step()
    #
    #             # 日志记录
    #             if self.batch_nums % self.log_every == 0:
    #                 logging.info(
    #                     f'Batch {self.batch_nums}: Loss={total_loss:.4f}, '
    #                     f'Main={loss_main:.4f}, Align={loss_soft_align:.4f}, '
    #                     f'Contrast={loss_contrastive:.4f}, '
    #                     f'Temp={self.matcher.temperature.item():.4f}'
    #                 )
    #
    #             self.batch_nums += 1
    #             if self.batch_nums >= self.max_batches:
    #                 break
    #
    #         # 验证
    #         if (epoch + 1) % (self.eval_every // 100) == 0:
    #             val_auc = self.eval_zero_shot(mode='dev')
    #             if val_auc > best_val_auc:
    #                 best_val_auc = val_auc
    #                 self.save(self.save_path + 'best_zeroshot_model')
    #                 logging.info(f'New best model saved! AUC: {val_auc:.4f}')
    # ################################新增结束##########################
    def eval_acc(self, mode='dev', meta=False):
        self.matcher.eval()
        symbol2id = self.symbol2id
        logging.info('EVALUATING ON %s DATA' % mode.upper())

        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        elif mode == 'test':
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test2_tasks.json'))

        rel2id = json.load(open(self.dataset + '/relation2ids'))
        rel2candidates = self.rel2candidates

        probas_pred = []
        ground_truth = []

        # 遍历每一个测试任务
        for query_ in tqdm(test_tasks.keys(), desc="Evaluating Tasks"):
            query_triples = test_tasks[query_]
            if not query_triples:
                continue

            # 1. 构造正样本：ID对（用于Matcher）和 字符串对（用于Neighbor Encoder）
            query_pairs_ids = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            # 2. 构造负样本：采样字符串名称
            candidates = rel2candidates[query_]
            false_names = []
            for triple in query_triples:
                e_h, rel, e_t = triple[0], triple[1], triple[2]
                while True:
                    noise = random.choice(candidates)
                    # 确保采样到的不是正样本
                    if (noise not in self.e1rel_e2[e_h + rel]) and noise != e_t:
                        break
                false_names.append(noise)

            # 3. 构造负样本的 ID 对（用于 Matcher）
            false_pairs_ids = [[symbol2id[query_triples[i][0]], symbol2id[false_names[i]]] for i in
                               range(len(query_triples))]

            # 4. 合并所有样本（Matcher 需要的是 ID 数组）
            all_pairs_ids = query_pairs_ids + false_pairs_ids

            # 5. 构造用于邻居编码器的字符串格式三元组（用于 Meta 模式下的 self.ent2id 查找）
            # 正样本部分
            all_triples_for_meta = [[triple[0], triple[2], rel2id[triple[1]]] for triple in query_triples]
            # 负样本部分
            for i in range(len(query_triples)):
                all_triples_for_meta.append([query_triples[i][0], false_names[i], rel2id[query_triples[i][1]]])

            # --- 分批推理过程 ---
            probas_pred_task = []
            test_size = self.batch_size * 20

            for i in range(0, len(all_pairs_ids), test_size):
                batch_pairs = torch.LongTensor(all_pairs_ids[i: i + test_size]).to(DEVICE)
                batch_triples = all_triples_for_meta[i: i + test_size]

                if meta:
                    # 此时 batch_triples 里的 t[0] 和 t[1] 都是字符串名称
                    query_left = [self.ent2id[t[0]] for t in batch_triples]
                    query_right = [self.ent2id[t[1]] for t in batch_triples]
                    query_meta = self.get_meta(query_left, query_right)

                    # 构造图数据 batch (DrugDataset 内部处理字符串到图的转换)
                    batch_data = DrugDataset(batch_triples)
                    loader = DrugDataLoader(batch_data, batch_size=len(batch_triples), shuffle=False)
                    query_batch = [t.to('cuda') for t in next(iter(loader))]

                    self.G_m.eval()
                    task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()

                    # 证据学习模式推理：直接获取期望概率
                    with torch.no_grad():
                        scores_prob, _ = self.matcher(task_emb, batch_pairs, None, query_meta, None, query_batch, None,
                                                      self.optim_VAE, is_eval=True, trainGAN=False)

                    probas_pred_task.append(scores_prob.detach().cpu().numpy())
                else:
                    # 非 Meta 模式处理 (如果需要的话)
                    pass

            if not probas_pred_task:
                continue

            # 合并该任务的所有 Batch 结果
            y_pred = np.concatenate(probas_pred_task)
            y_true = np.concatenate([np.ones(len(y_pred) // 2), np.zeros(len(y_pred) // 2)])

            probas_pred.append(y_pred)
            ground_truth.append(y_true)

        # --- 计算全局指标 ---
        if not probas_pred:
            logging.error("Evaluation list is empty!")
            return 0.0

        all_y_pred = np.concatenate(probas_pred)
        all_y_true = np.concatenate(ground_truth)

        acc, auroc, f1, pre, rec, int_ap, ap = do_compute_metrics(all_y_pred, all_y_true)
        logging.info(
            f'[{mode.upper()}] Global Metrics: ROC: {auroc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}')

        # 记录到 ExperimentRecorder
        metrics_dict = {'acc': acc, 'auroc': auroc, 'f1_score': f1, 'ap': ap}
        self.recorder.record_evaluation(mode, metrics_dict, is_best=False, batch_num=self.batch_nums)

        self.matcher.train()
        return auroc

    #     def eval_acc(self, mode='dev', meta=False):
    #         self.matcher.eval()
    #         symbol2id = self.symbol2id
    #         logging.info('EVALUATING ON %s DATA' % mode.upper())
    #         if mode == 'dev':
    #             test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
    #         elif mode == 'test':
    #             test_tasks = json.load(open(self.dataset + '/test_tasks.json'))
    #         else:
    #             test_tasks = json.load(open(self.dataset + '/test2_tasks.json'))
    #         rel2id = json.load(open(self.dataset + '/relation2ids'))
    #
    #         rel2candidates = self.rel2candidates
    #
    #         probas_pred = []
    #         ground_truth = []
    #
    #         for query_ in test_tasks.keys():
    #
    #             probas_pred_t = []
    #             ground_truth_t = []
    #             candidates = rel2candidates[query_]
    #             few = 0
    #
    #             query_triples = test_tasks[query_][few:]
    #             query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
    #
    #             false_pairs = []
    #             false_triples = []
    #             for triple in query_triples:
    #                 e_h = triple[0]
    #                 rel = triple[1]
    #                 e_t = triple[2]
    #                 while True:
    #                     noise = random.choice(candidates)
    #                     if (noise not in self.e1rel_e2[e_h + rel]) and noise != e_t:
    #                         break
    #                 false_triples.append([e_h, rel, noise])
    #                 false_pairs.append([symbol2id[e_h], symbol2id[noise]])
    #
    #             query_pairs.extend(false_pairs)
    #             query_triples.extend(false_triples)
    #             query_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in query_triples]
    #
    #             query = Variable(torch.LongTensor(query_pairs)).cuda()
    #
    #             test_size = self.batch_size * 800
    #             if len(query_triples_rel2id) < test_size:
    #                 test_size = len(query_triples_rel2id)
    #             # for i in range(len(query_triples_rel2id) // test_size):
    #             for i in range(0, len(query_triples_rel2id), test_size):#new
    #                 # if (i + 1) * test_size > len(query_triples_rel2id):
    #                 #     query_triples_rel2id_batch = query_triples_rel2id[i * test_size:]
    #                 # else:
    #                 #     query_triples_rel2id_batch = query_triples_rel2id[i * test_size: (i + 1) * test_size]
    #                 query_triples_rel2id_batch = query_triples_rel2id[i: i + test_size]#new
    #
    #
    #
    #
    #                 if meta:
    #                     query_left = [self.ent2id[triple[0]] for triple in query_triples]
    #                     query_right = [self.ent2id[triple[2]] for triple in query_triples]
    #                     query_meta = self.get_meta(query_left, query_right)
    #                     query_batch = DrugDataset(query_triples_rel2id)
    #                     query_batch_loader = DrugDataLoader(query_batch, batch_size=len(query_triples_rel2id),
    #                                                         shuffle=False)
    #                     query_batch = []
    #                     for batch in query_batch_loader:
    #                         query_batch.append(batch)
    #                     query_batch = [t.to(device) for t in query_batch[0]]
    #                     ############old#
    #                     # self.G_m.eval()
    #                     # task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()
    #                     # self.G_m.train()
    #                     # scores, loss2 = self.matcher(task_emb, query, None, query_meta, None, query_batch, None,
    #                     #                              self.optim_VAE, is_eval=True, trainGAN=False)
    #                     # scores.detach()
    #                     # scores = scores.data
    #                     # probas_pred_t.append(np.concatenate([torch.sigmoid(scores.detach()).cpu()]))
    #                     #new
    #                     self.G_m.eval()
    #                     task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()
    #
    #                     # 评估模式：返回 alpha / S 的概率
    #                     scores_prob, _ = self.matcher(task_emb, query, None, query_meta, None, query_batch, None,
    #                                                   self.optim_VAE, is_eval=True, trainGAN=False)
    #
    #                     # 此时 scores_prob 已经是概率，直接存入
    #                     probas_pred_t.append(scores_prob.detach().cpu().numpy())
    # #############################################
    #
    #
    #
    #                 else:
    #                     scores, loss2 = self.matcher(query, support)
    #                     scores.detach()
    #                     scores = scores.data
    #                     probas_pred_t.append(np.concatenate([torch.sigmoid(scores.detach()).cpu()]))
    #                     ##################new##############
    #                     # 这里的 probas_pred_t[0] 包含了 [正样本概率, 负样本概率]
    #                     num_total = len(probas_pred_t[0])
    #                     ground_truth_t.append(np.concatenate([np.ones(num_total // 2), np.zeros(num_total // 2)]))
    #
    #                     # 记录日志
    #                     acc, auroc, f1, pre, rec, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred_t),
    #                                                                               np.concatenate(ground_truth_t))
    #
    #
    #
    #
    # ##################old######################
    #             # ground_truth_t.append(
    #             #     np.concatenate([np.ones(int(len(probas_pred_t[0]) / 2)), np.zeros(int(len(probas_pred_t[0]) / 2))]))
    #             # loss, loss_p, loss_n = loss_fn(scores[:int(len(probas_pred_t[0]) / 2)],
    #             #                                scores[int(len(probas_pred_t[0]) / 2):])
    #             #
    #
    #
    #
    #
    #             acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred_t),
    #                                                                                      np.concatenate(ground_truth_t))
    #             logging.info(
    #                 f'task: {query_}\n loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')
    #             probas_pred.extend(probas_pred_t)
    #             ground_truth.extend(ground_truth_t)
    #
    #         acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred),
    #                                                                                  np.concatenate(ground_truth))
    #         logging.info(
    #             f'alltask:\n loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')
    #
    #         # 记录评估结果
    #         metrics_dict = {
    #             'acc': acc, 'auroc': auroc, 'f1_score': f1_score,
    #             'precision': precision, 'recall': recall, 'int_ap': int_ap, 'ap': ap
    #         }
    #         batch_num = getattr(self, 'batch_nums', None)
    #         self.recorder.record_evaluation(mode, metrics_dict, is_best=False, batch_num=batch_num)
    #
    #         self.matcher.train()
    #         return auroc

    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        testauc = self.eval_acc(meta=self.meta, mode='test')
        test2auc = self.eval_acc(meta=self.meta, mode='test2')
        # 完成测试记录
        self.recorder.finalize()


#######################old#######################
# class SigmoidLoss(nn.Module):
#
#     def forward(self, p_scores, n_scores):
#         p_loss = - F.logsigmoid(p_scores).mean()
#         n_loss = - F.logsigmoid(-n_scores).mean()
#
#         return (p_loss + n_loss) / 2, p_loss, n_loss
###########################new######################3
class EvidentialLoss(nn.Module):
    """
    EviDTI 核心损失函数：基于狄利克雷分布的深度证据学习损失
    """

    def __init__(self, annealing_step=500):
        super(EvidentialLoss, self).__init__()
        self.annealing_step = annealing_step  # 逐步增加 KL 散度的权重

    def kl_divergence(self, alpha):
        ones = torch.ones([1, 2], dtype=torch.float32).to(DEVICE)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + \
                     torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
        second_term = (alpha - ones).detach() * (torch.digamma(alpha) - torch.digamma(sum_alpha))
        return first_term + second_term.sum(dim=1, keepdim=True)

    def forward(self, alpha, target_type, batch_num):
        # target_type: 'pos' 为正样本, 'neg' 为负样本
        if target_type == 'pos':
            y = torch.tensor([[0, 1]], dtype=torch.float32).repeat(alpha.size(0), 1).to(DEVICE)
        else:
            y = torch.tensor([[1, 0]], dtype=torch.float32).repeat(alpha.size(0), 1).to(DEVICE)

        S = torch.sum(alpha, dim=1, keepdim=True)
        # 1. 计算 Expected Mean Square Error
        p = alpha / S
        err = (y - p) ** 2
        var = p * (1 - p) / (S + 1)
        loss_mse = torch.mean(torch.sum(err + var, dim=1))

        # 2. 计算 KL 散度正则项 (移除正确类别的证据后的惩罚)
        annealing_coef = min(1.0, batch_num / self.annealing_step)
        alpha_hat = y + (1 - y) * alpha
        loss_kl = annealing_coef * torch.mean(self.kl_divergence(alpha_hat))

        return loss_mse + loss_kl

    ################################################


if __name__ == '__main__':
    args = read_options()

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
    # loss_fn = SigmoidLoss()原始
    loss_fn = EvidentialLoss(annealing_step=10000)
    ###################new
    trainer = Trainer(args)
    if args.test:
        trainer.test_()
    else:
        trainer.train()