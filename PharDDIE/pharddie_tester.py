6#!/usr/bin/env Python
# coding=utf-8

from pharddie_args import read_options
from pharddie_dataloader import *
from pharddie_matcher import *
from tensorboardX import SummaryWriter
from sklearn import metrics
import csv
from shared.checkpoint import load_state_dict_safe


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


class Trainer(object):

    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)

        # 强制设置save_path为 models/intial（覆盖传入参数）
        self.save_path = 'models/intial'
        self.meta = not self.no_meta
        self.device = 'cpu'  # 固定使用CPU

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

        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols
        # 初始化模型并移至CPU（替换.cuda()）
        self.matcher = EmbedMatcher(
            self.embed_dim, self.num_symbols,
            use_pretrain=self.use_pretrain,
            embed=self.symbol2vec,
            dropout=self.dropout,
            batch_size=self.batch_size,
            finetune=self.fine_tune,
            aggregate=self.aggregate
        )
        self.matcher.to(self.device)  # 关键修改：移除.cuda()，改为CPU
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
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        # 加载答案字典
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        self.all_drug_data = {}
        self.drug_num_node_indices = {}

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
        with open(self.dataset + '/path_graph_train_only') as f:  # P0-5：ACI 只读取净化图
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

    def de(self, a):
        a.pop('support_encoder.proj1.weight', None)
        a.pop('support_encoder.proj1.bias', None)
        a.pop('support_encoder.proj2.weight', None)
        a.pop('support_encoder.proj2.bias', None)
        a.pop('support_encoder.layer_norm.a_2', None)
        a.pop('support_encoder.layer_norm.b_2', None)
        a.pop('query_encoder.process.weight_ih', None)
        a.pop('query_encoder.process.weight_hh', None)
        a.pop('query_encoder.process.bias_ih', None)
        a.pop('query_encoder.process.bias_hh', None)
        return a

    def save(self, path=None):
        if not path:
            path = self.save_path  # 使用固定的 save_path: models/intial
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.matcher.state_dict(), path)

    def load(self):
        # 加载模型时指定CPU
        self.matcher.load_state_dict(torch.load(self.save_path, map_location=self.device))

    def get_meta(self, left, right):
        # 所有变量移至CPU（替换.cuda()）
        left_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).to(
            self.device)
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        right_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).to(
            self.device)
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (left_connections, left_degrees, right_connections, right_degrees)

    def eval_acc(self, mode='dev', meta=False):
        self.matcher.eval()
        symbol2id = self.symbol2id
        few = self.few
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

        for query_ in test_tasks.keys():
            probas_pred_t = []
            ground_truth_t = []
            if len(test_tasks[query_]) < few + 1:
                continue

            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
            support_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in support_triples]

            support_batch = DrugDataset(support_triples_rel2id)
            support_batch_loader = DrugDataLoader(support_batch, batch_size=len(support_triples_rel2id), shuffle=False)
            support_batch = [batch for batch in support_batch_loader]

            support_left = [self.ent2id[triple[0]] for triple in support_triples]
            support_right = [self.ent2id[triple[2]] for triple in support_triples]
            support_meta = self.get_meta(support_left, support_right)

            # 变量移至CPU（替换.cuda()）
            support = Variable(torch.LongTensor(support_pairs)).to(self.device)

            query_triples = test_tasks[query_][few:]
            query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            false_pairs = []
            false_triples = []
            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if (noise not in self.e1rel_e2[e_h + rel]) and noise != e_t:
                        break
                false_triples.append([e_h, rel, noise])
                false_pairs.append([symbol2id[e_h], symbol2id[noise]])

            query_pairs.extend(false_pairs)
            query_triples.extend(false_triples)
            query_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in query_triples]

            # 变量移至CPU（替换.cuda()）
            query = Variable(torch.LongTensor(query_pairs)).to(self.device)

            test_size = self.batch_size * 800
            if len(query_triples_rel2id) < test_size:
                test_size = len(query_triples_rel2id)

            for i in range(len(query_triples_rel2id) // test_size + 1):
                start = i * test_size
                end = start + test_size
                query_triples_rel2id_batch = query_triples_rel2id[start:end]
                if not query_triples_rel2id_batch:
                    continue

                query_left = [self.ent2id[triple[0]] for triple in query_triples]
                query_right = [self.ent2id[triple[2]] for triple in query_triples]
                query_meta = self.get_meta(query_left, query_right)

                query_batch = DrugDataset(query_triples_rel2id_batch)
                query_batch_loader = DrugDataLoader(query_batch, batch_size=len(query_triples_rel2id_batch),
                                                    shuffle=False)
                query_batch = [batch for batch in query_batch_loader]

                # 数据移至CPU（替换.to(device)，device原为cuda）
                support_batch_cpu = [t.to(self.device) for t in support_batch[0]] if support_batch else []
                query_batch_cpu = [t.to(self.device) for t in query_batch[0]] if query_batch else []

                scores, loss2 = self.matcher(query, support, query_meta, support_meta, query_batch_cpu,
                                             support_batch_cpu, self.optim_VAE)
                scores = scores.detach()
                probas_pred_t.append(torch.sigmoid(scores).cpu().numpy())

            ground_truth_t.append(np.concatenate([
                np.ones(int(len(probas_pred_t[0]) / 2)),
                np.zeros(int(len(probas_pred_t[0]) / 2))
            ]))

            loss, loss_p, loss_n = loss_fn(scores[:int(len(probas_pred_t[0]) / 2)],
                                           scores[int(len(probas_pred_t[0]) / 2):])
            acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(
                np.concatenate(probas_pred_t),
                np.concatenate(ground_truth_t)
            )
            logging.info(
                f'task: {query_}\n loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')
            probas_pred.extend(probas_pred_t)
            ground_truth.extend(ground_truth_t)

        acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(
            np.concatenate(probas_pred),
            np.concatenate(ground_truth)
        )
        logging.info(
            f'alltask:\n loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')

        # 确保results文件夹存在
        os.makedirs('results', exist_ok=True)
        with open(
                f'results/{args.dataset}_{args.few}shot_{mode}_acc{acc:.4f}_auroc{auroc:.4f}_f1_score{f1_score:.4f}.csv',
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['acc', 'auroc', 'f1_score'])
            writer.writerow([f'{acc:.4f}', f'{auroc:.4f}', f'{f1_score:.4f}'])

        all_results.append([acc, auroc, f1_score, args.few, mode])
        self.matcher.train()
        return auroc

    def test_(self, model_P):
        # 加载模型时指定CPU，并处理参数兼容
        model_state = torch.load(model_P, map_location=self.device)
        model_state = self.de(model_state)  # 移除不兼容的参数键
        load_state_dict_safe(self.matcher, model_state, model_name='matcher')
        logging.info('Pre-trained model loaded from: %s' % model_P)
        self.eval_acc(mode='test', meta=self.meta)
        self.eval_acc(mode='test2', meta=self.meta)


class SigmoidLoss(nn.Module):
    def forward(self, p_scores, n_scores):
        p_loss = -F.logsigmoid(p_scores).mean()
        n_loss = -F.logsigmoid(-n_scores).mean()
        return (p_loss + n_loss) / 2, p_loss, n_loss


if __name__ == '__main__':
    args = read_options()

    # 日志配置
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

    # 随机种子（移除CUDA相关）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设备设置为CPU（覆盖原cuda）
    device = 'cpu'
    loss_fn = SigmoidLoss()

    all_results = []
    all_results.append(['acc', 'auroc', 'f1_score', 'few', 'test_n'])

    # 确保results文件夹存在
    os.makedirs('results', exist_ok=True)

    # 测试不同few-shot设置（使用dataset1，加载对应模型）
    for few in [1, 5, 10]:
        args.few = few
        args.train_few = few
        args.dataset = 'dataset1'
        trainer = Trainer(args)

        # 加载对应few-shot的模型（路径按你的实际文件修改）
        model_paths = {
            1: 'models/dataset1/models_drugbank_1shot_str/bestmodel',
            5: 'models/dataset1/models_drugbank_5shot_str/bestmodel',
            10: 'models/dataset1/models_drugbank_10shot_str/bestmodel'
        }
        trainer.test_(model_paths[few])


    # 保存汇总结果（save_path已固定为models/intial，但结果文件按dataset1命名）
    def StoreFile2(data, fileName):
        with open(fileName, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)


    StoreFile2(all_results, f'results/{args.dataset}_allresults.csv')