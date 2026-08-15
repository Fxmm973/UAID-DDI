#!/usr/bin/env Python
# coding=utf-8
"""
快速训练 w/o uncertainty 变体：
冻结 PharDDIE 的分子编码器 + 邻居编码器，
只训练一个新的 fc 头（无 VAE），用邻居嵌入直接做距离分类。

训练时间：每个 shot 约 2-5 分钟（1000 iterations × 快速收敛）
"""
import torch.nn as nn, torch.nn.functional as F
from collections import deque
from torch import optim
from torch.autograd import Variable
from pharddie_args import read_options
from pharddie_dataloader import *
from pharddie_matcher import EmbedMatcher
from shared.checkpoint import load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class SigmoidLoss(nn.Module):
    def forward(self, p_scores, n_scores):
        p_loss = -F.logsigmoid(p_scores).mean()
        n_loss = -F.logsigmoid(-n_scores).mean()
        return (p_loss + n_loss) / 2, p_loss, n_loss

class Trainer(object):
    def __init__(self, arg):
        for k, v in vars(arg).items():
            setattr(self, k, v)
        self.meta = not self.no_meta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载符号表
        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols

        # 加载预训练 PharDDIE（完整模型）
        self.matcher = EmbedMatcher(
            self.embed_dim, self.num_symbols,
            use_pretrain=True, embed=self.symbol2vec,
            dropout=self.dropout, batch_size=self.batch_size,
            finetune=self.fine_tune, aggregate=self.aggregate
        ).to(self.device)

        # 加载预训练权重
        ckpt = torch.load(arg.pretrained_model, map_location=self.device)
        # 移除不兼容的键
        for k in list(ckpt.keys()):
            if any(x in k for x in ['support_encoder.proj', 'support_encoder.layer_norm',
                                     'query_encoder.process', 'fc_struc_net']):
                del ckpt[k]
        load_state_dict_safe(self.matcher, ckpt, model_name='matcher')
        logging.info(f'Loaded pretrained model from {arg.pretrained_model}')

        # 冻结所有参数
        for p in self.matcher.parameters():
            p.requires_grad = False

        # ---- 新建 w/o uncertainty 头 ----
        # 邻居嵌入维度: embed_dim*2 = 256
        neighbor_dim = self.embed_dim * 2
        self.fc_direct = nn.Sequential(
            nn.Linear(neighbor_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.batch_nums = 0
        self.optim = optim.Adam(self.fc_direct.parameters(), lr=0.001, weight_decay=0.0)

        # 数据
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))
        self.all_drug_data = {}
        self.drug_num_node_indices = {}

    def load_embed(self):
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        relation2embids = json.load(open(self.dataset + '/relation2embids'))
        ent2embids = json.load(open(self.dataset + '/ent2embids'))
        ent_embed = np.load(self.dataset + '/DRKG_TransE_entity.npy')
        rel_embed = np.load(self.dataset + '/DRKG_TransE_relation.npy')
        i = 0
        embeddings = []
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
        symbol_id['PAD'] = i
        embeddings.append(list(np.zeros((rel_embed.shape[1],))))
        self.symbol2id = symbol_id
        self.symbol2vec = np.array(embeddings)

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph_train_only') as f:  # P0-5：ACI 只读取净化图
            for line in tqdm(f.readlines(), desc='Building connections'):
                e1, rel, e2 = line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel], self.symbol2id[e2]))
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                random.shuffle(neighbors)
                neighbors = neighbors[:max_]
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

    def get_meta(self, left, right):
        lc = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).to(self.device)
        ld = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        rc = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).to(self.device)
        rd = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (lc, ld, rc, rd)

    def get_neighbor_embedding(self, pairs, meta):
        """获取邻居嵌入（无 VAE）"""
        lc, ld, rc, rd = meta
        ql_, qr_ = self.matcher.model(pairs)
        ql = self.matcher.neighbor_encoder(lc, ld, ql_, qr_ - ql_)
        qr = self.matcher.neighbor_encoder(rc, rd, qr_, qr_ - ql_)
        return torch.cat((ql, qr), dim=-1)  # [batch, 256]

    def train_quick(self):
        logging.info(f'Training w/o uncertainty head for {self.max_batches} iterations...')
        loss_fn = SigmoidLoss()
        losses = deque([], 50)

        for data in train_generate(self.dataset, self.batch_size, self.train_few,
                                   self.symbol2id, self.ent2id, self.e1rel_e2,
                                   self.all_drug_data, self.drug_num_node_indices):
            support, query, false, sl, sr, ql, qr, fl, fr, sb, qb, fb = data
            sb = [t.to(self.device) for t in sb]
            qb = [t.to(self.device) for t in qb]
            fb = [t.to(self.device) for t in fb]

            s_meta = self.get_meta(sl, sr)
            q_meta = self.get_meta(ql, qr)
            f_meta = self.get_meta(fl, fr)

            with torch.no_grad():
                s_emb = self.get_neighbor_embedding(sb, s_meta)
                q_emb = self.get_neighbor_embedding(qb, q_meta)
                f_emb = self.get_neighbor_embedding(fb, f_meta)

            # 距离计算（无 VAE）
            s_mean = s_emb.mean(dim=0, keepdim=True)
            q_scores = self.fc_direct(torch.abs(s_mean.expand_as(q_emb) - q_emb))
            f_scores = self.fc_direct(torch.abs(s_mean.expand_as(f_emb) - f_emb))

            loss, _, _ = loss_fn(q_scores, f_scores)
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (self.batch_nums + 1) % 200 == 0:
                logging.info(f'  iter {self.batch_nums+1}/{self.max_batches}, loss={np.mean(losses):.4f}')

            self.batch_nums += 1
            if self.batch_nums >= self.max_batches:
                break

        # 保存
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.fc_direct.state_dict(), self.save_path)
        logging.info(f'Saved w/o uncertainty head to {self.save_path}')

if __name__ == '__main__':
    args = read_options()
    args.max_batches = 5000  # 快速训练
    args.batch_size = 256
    args.train_few = args.few
    args.dataset = 'dataset1'

    # 3 shots × 各自的预训练模型
    for few in [1, 5, 10]:
        args.few = few
        args.train_few = few
        args.pretrained_model = f'models/dataset1/models_drugbank_{few}shot_str/bestmodel'
        args.save_path = f'models/dataset1/models_wo_uncertainty_{few}shot/bestmodel'
        args.seed = 2024  # 单 seed 快速训练

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        trainer = Trainer(args)
        trainer.train_quick()

    logging.info('All done! 3 models saved.')
