#!/usr/bin/env Python
# coding=utf-8
"""
DEPRECATED (P0-7 cleanup): superseded by eviddie_trainer.py (formal zero-shot
entry with a native dual-output EDL head) and eviddie_train_ablation.py
(frozen-backbone head ablation, Figure 4). Kept for reference only; do not
use to produce paper results.
"""
import torch.nn as nn, torch.nn.functional as F
import csv
from collections import deque
from torch import optim
from torch.autograd import Variable
from eviddie_args import read_options
from eviddie_dataloader import *
from eviddie_matcher import EmbedMatcher, Generate_Model
from sklearn import metrics
from shared.checkpoint import load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

MAX_ITER = 10000
EVAL_EVERY = 500


def comp_metrics(probas, targets):
    if len(np.unique(targets)) < 2: return 0.5, 0.5, 0.5
    pred = (probas >= 0.5).astype(int)
    return (metrics.accuracy_score(targets, pred),
            metrics.roc_auc_score(targets, probas),
            metrics.f1_score(targets, pred, zero_division=0))


class QuickTrainer(object):
    def __init__(self, arg):
        for k, v in vars(arg).items(): setattr(self, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device}")

        self.semantic_task = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        for task in list(self.semantic_task.keys()):
            self.semantic_task[task] = np.array(self.semantic_task[task]) + \
                0.3 * np.random.normal(loc=0, scale=1, size=(len(self.semantic_task[task]), 1))
        self.task_ebmedding = []
        self.task2id = {}
        for num, i in enumerate(list(self.semantic_task.keys())):
            self.task2id[i] = num
            self.task_ebmedding.append(self.semantic_task[i])
        self.task_ebmedding = torch.tensor(np.vstack(self.task_ebmedding)).float().to(self.device)
        self.use_pretrain = not self.random_embed

        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols

        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols,
                                     use_pretrain=self.use_pretrain, embed=self.symbol2vec,
                                     dropout=self.dropout, batch_size=self.batch_size,
                                     finetune=self.fine_tune, aggregate=self.aggregate,
                                     task_emb=self.task_ebmedding).to(self.device)

        ckpt = torch.load(arg.pretrained_model, map_location=self.device)
        if 'fc.5.weight' in ckpt and ckpt['fc.5.weight'].shape[0] == 1:
            logging.info('Converting old 1-output fc to EDL 2-output...')
            ow, ob = ckpt['fc.5.weight'], ckpt['fc.5.bias']
            ckpt['fc.5.weight'] = torch.cat([ow, -ow], dim=0)
            ckpt['fc.5.bias'] = torch.cat([ob, -ob], dim=0)
        for k in list(ckpt.keys()):
            if any(x in k for x in ['symbol_emb','gcn_w','gcn_b','Bilinear','Linear_self',
                                     'Linear_nei','Linear_weak_rel','NeighborAggregator','siamese',
                                     'support_encoder','query_encoder']):
                del ckpt[k]
        load_state_dict_safe(self.matcher, ckpt, model_name='matcher')
        logging.info(f'Loaded pretrained encoder from {arg.pretrained_model}')

        for n, p in self.matcher.named_parameters():
            if 'fc' not in n:
                p.requires_grad = False
        logging.info('Encoder frozen, only fc head trainable')

        self.G_m = Generate_Model(in_dim=self.task_ebmedding.shape[1]).to(self.device)
        self.G_m = torch.load(arg.g_model_path, map_location=self.device)
        self.G_m.eval()

        # w/o BSA: linear projection 替代 GAN generator
        self.linear_proj = nn.Linear(self.task_ebmedding.shape[1], 64).to(self.device)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))
        self.all_drug_data = {}
        self.drug_num_node_indices = {}

        # Dev tasks for eval
        self.dev_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))

        self.loss_fn = SigmoidLoss()

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

    def encode_pair(self, pairs_batch):
        """CSE + VAE latent for a drug pair batch"""
        ql_, qr_ = self.matcher.model(pairs_batch)
        qn = torch.cat((ql_, qr_), dim=-1)
        _, _, _, zq = self.matcher.vaemodel(qn, is_support=False, is_eval=True)
        return zq  # [batch, 64]

    def get_task_proto(self, task_name, use_gan=True):
        with torch.no_grad():
            if use_gan:
                return self.G_m(self.task_ebmedding[self.task2id[task_name]]).detach()
            else:
                return self.linear_proj(self.task_ebmedding[self.task2id[task_name]].unsqueeze(0)).detach()

    def eval_dev(self, variant_name):
        self.matcher.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for query_, triples in self.dev_tasks.items():
                if not triples: continue
                cand = self.rel2candidates[query_]
                np.random.seed(hash(query_) % 100000 + 42)
                ft = []
                for t in triples:
                    eh, rel, et = t[0], t[1], t[2]
                    while True:
                        n = np.random.choice(cand)
                        if n not in self.e1rel_e2.get(eh+rel, []) and n != et: break
                    ft.append([eh, rel, n])
                at = triples + ft
                ar = [[t[0], t[2], self.rel2id[t[1]]] for t in at]
                npos = len(triples)
                qb = DrugDataset(ar)
                qbl = DrugDataLoader(qb, batch_size=len(ar), shuffle=False)
                qbd = [t.to(self.device) for t in next(iter(qbl))]
                use_gan = (variant_name != 'w/o BSA')
                proto = self.get_task_proto(query_, use_gan=use_gan)
                zq = self.encode_pair(qbd)
                fc_out = self.matcher.fc(torch.abs(proto.expand_as(zq) - zq))
                if variant_name == 'softmax':
                    prob = F.softmax(fc_out, dim=1)[:, 1]
                else:
                    al = F.softplus(fc_out) + 1; prob = al[:, 1] / al.sum(1)
                all_p.append(prob.cpu().numpy())
                all_l.append(np.concatenate([np.ones(npos), np.zeros(len(at)-npos)]))
        return comp_metrics(np.concatenate(all_p), np.concatenate(all_l))

    def train_variant(self, variant_name, csv_writer, max_iter=MAX_ITER):
        """训练一个 fc 变体"""
        logging.info(f'=== Training {variant_name} ({max_iter} iters) ===')
        params = list(self.matcher.fc.parameters())
        if variant_name == 'w/o BSA':
            params += list(self.linear_proj.parameters())
        optimizer = optim.Adam(params, lr=0.001, weight_decay=0.0)
        losses = deque([], 50)
        batch_nums = 0
        use_gan = (variant_name != 'w/o BSA')

        for data in train_generate(self.dataset, self.batch_size, self.train_few,
                                     self.symbol2id, self.ent2id, self.e1rel_e2,
                                     self.all_drug_data, self.drug_num_node_indices):

            task_name, support, query, false, sl, sr, ql, qr, fl, fr, sb, qb, fb = data
            qb = [t.to(self.device) for t in qb]
            fb = [t.to(self.device) for t in fb]

            task_proto = self.get_task_proto(task_name, use_gan=use_gan)

            with torch.no_grad():
                zq = self.encode_pair(qb)
                zf = self.encode_pair(fb)

            q_out = self.matcher.fc(torch.abs(task_proto.expand_as(zq) - zq))
            f_out = self.matcher.fc(torch.abs(task_proto.expand_as(zf) - zf))

            if variant_name == 'softmax':
                pos_logits = q_out[:, 1] - q_out[:, 0]
                neg_logits = f_out[:, 1] - f_out[:, 0]
                loss, _, _ = self.loss_fn(pos_logits, neg_logits)

            elif variant_name in ('evi_no_evi', 'w/o BSA'):
                evidence_q = F.softplus(q_out); alpha_q = evidence_q + 1
                prob_q = alpha_q[:, 1] / alpha_q.sum(dim=1)
                evidence_f = F.softplus(f_out); alpha_f = evidence_f + 1
                prob_f = alpha_f[:, 1] / alpha_f.sum(dim=1)
                loss = F.mse_loss(prob_q, torch.ones_like(prob_q)) + \
                       F.mse_loss(prob_f, torch.zeros_like(prob_f))

            else:  # 'full_evi'
                evidence_q = F.softplus(q_out); alpha_q = evidence_q + 1; S_q = alpha_q.sum(dim=1)
                evidence_f = F.softplus(f_out); alpha_f = evidence_f + 1; S_f = alpha_f.sum(dim=1)
                mse_q = F.mse_loss(alpha_q[:,1]/S_q, torch.ones_like(alpha_q[:,1]))
                mse_f = F.mse_loss(alpha_f[:,1]/S_f, torch.zeros_like(alpha_f[:,1]))
                def kl_divergence(alpha, S):
                    K = alpha.shape[1]; alpha0 = torch.ones_like(alpha)
                    kl = torch.lgamma(S) - torch.lgamma(torch.tensor(K, dtype=S.dtype, device=S.device)) \
                         - (torch.lgamma(alpha) - torch.lgamma(alpha0)).sum(dim=1) \
                         + ((alpha - alpha0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
                    return kl.mean()
                kl = kl_divergence(alpha_q, S_q) + kl_divergence(alpha_f, S_f)
                loss = 0.5 * (mse_q + mse_f) + 0.005 * kl

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_nums += 1

            if batch_nums % EVAL_EVERY == 0 or batch_nums == 1:
                self.matcher.eval()
                acc, auroc, f1 = self.eval_dev(variant_name)
                self.matcher.train()
                logging.info(f'  [{variant_name}] iter={batch_nums}/{max_iter} loss={np.mean(losses):.4f} dev_auroc={auroc:.4f} dev_f1={f1:.4f}')
                csv_writer.writerow([variant_name, batch_nums, np.mean(losses), auroc, f1, acc])

            if batch_nums >= max_iter:
                break

        save_path = f'{self.save_dir}/fc_{variant_name}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.matcher.fc.state_dict(), save_path)
        if variant_name == 'w/o BSA':
            torch.save(self.linear_proj.state_dict(), f'{self.save_dir}/linear_proj_wo_BSA.pt')
        logging.info(f'  Saved {variant_name} head to {save_path}')


class SigmoidLoss(nn.Module):
    def forward(self, p_scores, n_scores):
        p_loss = -F.logsigmoid(p_scores).mean()
        n_loss = -F.logsigmoid(-n_scores).mean()
        return (p_loss + n_loss) / 2, p_loss, n_loss


if __name__ == '__main__':
    args = read_options()
    args.dataset = 'dataset1'
    args.max_batches = MAX_ITER
    args.batch_size = 256
    args.train_few = 10
    args.no_meta = False
    args.random_embed = False
    args.pretrained_model = 'models/dataset1/bestmodels'
    args.g_model_path = 'models/dataset1/bestmodels_G'
    args.save_dir = 'models/dataset1'

    random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(2024)

    os.makedirs('results', exist_ok=True)
    f = open('results/ablation_curves.csv', 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['variant', 'iter', 'train_loss', 'dev_auroc', 'dev_f1', 'dev_acc'])

    trainer = QuickTrainer(args)

    for variant in ['softmax', 'w/o BSA', 'evi_no_evi', 'full_evi']:
        for layer in trainer.matcher.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if variant == 'w/o BSA':
            trainer.linear_proj.reset_parameters()
        trainer.train_variant(variant, w, max_iter=MAX_ITER)

    f.close()
    logging.info(f'Done -> results/ablation_curves.csv')
