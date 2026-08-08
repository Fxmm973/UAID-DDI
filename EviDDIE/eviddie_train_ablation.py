#!/usr/bin/env python
"""EviDDIE 消融训练 v3：直接用 PharDDIE matcher 加载预训练编码器，不经过 EviDDIE matcher"""
import json, logging, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, os, sys, csv
from collections import defaultdict, deque
from torch import optim
from sklearn import metrics
# Root path for checkpoint_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# PharDDIE paths for matcher and args (lower priority than EviDDIE local)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PharDDIE'))
from eviddie_args import read_options
from pharddie_matcher import EmbedMatcher as PharDDIEMatcher, VAE as SRAE
from shared.checkpoint import load_state_dict_safe

# EviDDIE local LAST = highest priority for data_loader (has task_name in yield)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eviddie_dataloader import DrugDataset, DrugDataLoader, train_generate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ITER = 5000
EVAL_EVERY = 200
SEED = 2024


def compute_metrics(probas, targets):
    if len(np.unique(targets)) < 2: return 0.5, 0.5, 0.5
    pred = (probas >= 0.5).astype(int)
    return (metrics.accuracy_score(targets, pred),
            metrics.roc_auc_score(targets, probas),
            metrics.f1_score(targets, pred, zero_division=0))


class EviDDIEAblationTrainer:
    """直接加载 PharDDIE checkpoint，在 SRAE latent 空间上训练 EviDDIE 的消融 fc 头。"""
    def __init__(self, dataset='dataset1', ckpt_path='models/dataset1/pharddie_best.pt'):
        # ---------- PharDDIE matcher (加载预训练权重) ----------
        # 先用 train=False 构造 PharDDIEMatcher，但不加载任何 KB embedding
        # 我们只需要它的 model (MVN_DDI) + vaemodel (SRAE)
        self.embed_dim = 128
        # 直接构造核心模块
        n_atom_feats, kge_dim = 55, 128
        from pharddie_matcher import MVN_DDI
        self.molecular_encoder = MVN_DDI(
            [n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, 0, [64, 64], [2, 2], 64, 0.0
        ).to(device)
        self.srae = SRAE(emb_dim=kge_dim * 2).to(device)

        # 加载预训练权重
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            # 提取 encoder 和 srae 的权重
            encoder_state = {}
            srae_state = {}
            for k, v in ckpt.items():
                if k.startswith('model.'):
                    encoder_state[k.replace('model.', '')] = v
                elif k.startswith('vaemodel.'):
                    srae_state[k.replace('vaemodel.', '')] = v
            if encoder_state:
                load_state_dict_safe(self.molecular_encoder, encoder_state, model_name='molecular_encoder')
                logging.info(f'Loaded encoder: {len(encoder_state)} keys')
            if srae_state:
                load_state_dict_safe(self.srae, srae_state, model_name='srae')
                logging.info(f'Loaded SRAE: {len(srae_state)} keys')
        else:
            raise FileNotFoundError(
                f'Checkpoint not found: {ckpt_path}. '
                f'Ablation experiments require a valid pretrained PharDDIE checkpoint '
                f'to ensure the frozen encoder and SRAE are meaningful. '
                f'Please train PharDDIE first or place the checkpoint at {ckpt_path}.'
            )

        # Freeze encoder + SRAE
        for p in self.molecular_encoder.parameters(): p.requires_grad = False
        for p in self.srae.parameters(): p.requires_grad = False
        self.molecular_encoder.eval()
        self.srae.eval()

        # ---------- G_m (GAN generator) ----------
        gm_path = 'models/dataset1/bestmodels_G'
        gm_loaded = False
        if os.path.exists(gm_path):
            gm_state = torch.load(gm_path, map_location=device)
            if isinstance(gm_state, dict) and 'fc.0.weight' in gm_state:
                gm_state = {k.replace('fc.', ''): v for k, v in gm_state.items()}
            elif hasattr(gm_state, 'state_dict'):
                gm_state = gm_state.state_dict()
                if 'fc.0.weight' in gm_state:
                    gm_state = {k.replace('fc.', ''): v for k, v in gm_state.items()}
            # Auto-detect sem_dim from checkpoint first layer
            if '0.weight' in gm_state:
                self.sem_dim = gm_state['0.weight'].shape[1]
            else:
                self.sem_dim = 768
            self.G_m = nn.Sequential(
                nn.Linear(self.sem_dim, 256), nn.Tanh(),
                nn.Linear(256, 512), nn.Tanh(),
                nn.Linear(512, 64), nn.Tanh(),
            ).to(device)
            try:
                self.G_m.load_state_dict(gm_state)
                gm_loaded = True
                logging.info('G_m loaded (sem_dim=%d)', self.sem_dim)
            except Exception as e:
                raise RuntimeError(
                    f'G_m (GAN generator) checkpoint failed to load: {e}. '
                    f'The ablation study requires a valid pretrained G_m to produce '
                    f'meaningful task prototypes for the BSA-based variants. '
                    f'Without it, "w/o BSA vs. full" comparison is invalid. '
                    f'Please verify the checkpoint at {gm_path}.'
                ) from e
        if not gm_loaded:
            self.sem_dim = 768
            self.G_m = nn.Sequential(
                nn.Linear(self.sem_dim, 256), nn.Tanh(),
                nn.Linear(256, 512), nn.Tanh(),
                nn.Linear(512, 64), nn.Tanh(),
            ).to(device)
        for p in self.G_m.parameters(): p.requires_grad = False
        self.G_m.eval()

        # w/o BSA 用线性投影替代 G_m
        self.linear_proj = nn.Linear(self.sem_dim, 64).to(device)

        # ---------- FC head (trainable) ----------
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        ).to(device)

        # ---------- 数据 ----------
        self.dataset = dataset
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._load_data()

    def _load_data(self):
        self.ds = os.path.join(self.base_dir, self.dataset)
        self.ent2id = json.load(open(os.path.join(self.ds, 'ent2ids')))
        self.rel2candidates = json.load(open(os.path.join(self.ds, 'rel2candidates.json')))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2.update(json.load(open(os.path.join(self.ds, 'e1rel_e2.json'))))
        self.rel2id = json.load(open(os.path.join(self.ds, 'relation2ids')))
        self.eval_tasks = json.load(open(os.path.join(self.ds, 'dev_tasks.json')))
        self.task_emb = json.load(open(os.path.join(self.ds, 'event_embedding2.json')))
        self.all_drug_data = {}
        self.drug_num_node_indices = {}

    def encode_pair(self, pairs_batch):
        """pairs_batch: DrugDataset batch on device → SRAE latent [B, 64]"""
        hl, hr = self.molecular_encoder(pairs_batch)
        qn = torch.cat((hl, hr), dim=-1)
        _, _, _, z = self.srae(qn, is_support=False, is_eval=True)
        return z

    def _get_task_emb(self, task_name):
        """Load task embedding and ensure shape [1, sem_dim]."""
        emb = np.array(self.task_emb[task_name], dtype=np.float32)
        emb = emb.reshape(-1)  # flatten to 1D
        return torch.tensor(emb, device=device).unsqueeze(0)

    def get_proto_gan(self, task_name):
        return self.G_m(self._get_task_emb(task_name)).detach()

    def get_proto_linear(self, task_name):
        return self.linear_proj(self._get_task_emb(task_name)).detach()

    def train_variant(self, variant_name, csv_writer, max_iter=MAX_ITER):
        logging.info(f'\n{"="*60}\nTraining {variant_name} ({max_iter} iters)\n{"="*60}')
        params = list(self.fc.parameters())
        if variant_name == 'w/o BSA':
            params += list(self.linear_proj.parameters())
            self.linear_proj.train()
        optimizer = optim.Adam(params, lr=0.001, weight_decay=0)
        losses = deque([], 50)
        step = 0

        for data in train_generate(self.ds, 256, 10,
                                     self.symbol2id, self.ent2id, self.e1rel_e2,
                                     self.all_drug_data, self.drug_num_node_indices):
            task_name, support, query, false = data[0], data[1], data[2], data[3]
            sl, sr, ql, qr, fl, fr = data[4], data[5], data[6], data[7], data[8], data[9]
            sb, qb, fb = data[10], data[11], data[12]
            sb = [t.to(device) for t in sb]
            qb = [t.to(device) for t in qb]
            fb = [t.to(device) for t in fb]

            proto = self.get_proto_gan(task_name) if variant_name != 'w/o BSA' else self.get_proto_linear(task_name)
            self.fc.train()

            zq = self.encode_pair(qb)
            zf = self.encode_pair(fb)
            q_out = self.fc(torch.abs(proto.expand_as(zq) - zq))
            f_out = self.fc(torch.abs(proto.expand_as(zf) - zf))

            if variant_name == 'softmax':
                loss = F.cross_entropy(q_out, torch.ones(q_out.size(0), dtype=torch.long, device=device)) + \
                       F.cross_entropy(f_out, torch.zeros(f_out.size(0), dtype=torch.long, device=device))
            elif variant_name in ('evi_no_evi', 'w/o BSA'):
                ev_q, al_q = F.softplus(q_out), F.softplus(q_out) + 1
                ev_f, al_f = F.softplus(f_out), F.softplus(f_out) + 1
                loss = F.mse_loss(al_q[:,1]/al_q.sum(1), torch.ones_like(al_q[:,1])) + \
                       F.mse_loss(al_f[:,1]/al_f.sum(1), torch.zeros_like(al_f[:,1]))
            else:
                ev_q, al_q = F.softplus(q_out), F.softplus(q_out) + 1
                ev_f, al_f = F.softplus(f_out), F.softplus(f_out) + 1
                Sq, Sf = al_q.sum(1), al_f.sum(1)
                mse = F.mse_loss(al_q[:,1]/Sq, torch.ones_like(al_q[:,1])) + \
                      F.mse_loss(al_f[:,1]/Sf, torch.zeros_like(al_f[:,1]))
                def kl(alpha, S):
                    K=alpha.shape[1]; a0=torch.ones_like(alpha)
                    return (torch.lgamma(S)-torch.lgamma(torch.tensor(float(K),device=device))
                            -(torch.lgamma(alpha)-torch.lgamma(a0)).sum(1)
                            +((alpha-a0)*(torch.digamma(alpha)-torch.digamma(S.unsqueeze(1)))).sum(1)).mean()
                loss = 0.5*mse + 0.005*(kl(al_q,Sq)+kl(al_f,Sf))

            losses.append(loss.item())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == 1:
                self.fc.eval()
                if variant_name == 'w/o BSA': self.linear_proj.eval()
                acc, auroc, f1 = self._eval_dev(variant_name)
                self.fc.train()
                if variant_name == 'w/o BSA': self.linear_proj.train()
                logging.info(f'  [{variant_name}] step={step}/{max_iter} loss={np.mean(losses):.4f} au={auroc:.4f} f1={f1:.4f} acc={acc:.4f}')
                csv_writer.writerow([variant_name, step, np.mean(losses), auroc, f1, acc])

            if step >= max_iter: break

        save_dir = os.path.join(self.base_dir, 'models', 'dataset1')
        os.makedirs(save_dir, exist_ok=True)
        save = os.path.join(save_dir, f'fc_{variant_name}.pt')
        torch.save(self.fc.state_dict(), save)
        if variant_name == 'w/o BSA':
            torch.save(self.linear_proj.state_dict(), os.path.join(save_dir, 'linear_proj_wo_BSA.pt'))
        logging.info(f'Saved {variant_name} → {save}')

    def _eval_dev(self, variant_name):
        all_p, all_l = [], []
        with torch.no_grad():
            for query_, triples in self.eval_tasks.items():
                if not triples: continue
                candidates = self.rel2candidates[query_]
                np.random.seed(hash(query_) % 100000 + 42)
                false_triples = []
                for t in triples:
                    eh, rel, et = t[0], t[1], t[2]
                    while True:
                        n = np.random.choice(candidates)
                        if n not in self.e1rel_e2.get(eh+rel,[]) and n!=et: break
                    false_triples.append([eh, rel, n])
                at = triples + false_triples
                ar = [[t[0], t[2], self.rel2id[t[1]]] for t in at]
                npos = len(triples)
                qb = DrugDataset(ar)
                qbl = DrugDataLoader(qb, batch_size=len(ar), shuffle=False)
                qbd = [t.to(device) for t in next(iter(qbl))]
                proto = self.get_proto_gan(query_) if variant_name != 'w/o BSA' else self.get_proto_linear(query_)
                zq = self.encode_pair(qbd)
                fc_out = self.fc(torch.abs(proto.expand_as(zq) - zq))
                if variant_name == 'softmax':
                    prob = F.softmax(fc_out, dim=1)[:,1]
                else:
                    al = F.softplus(fc_out)+1; prob = al[:,1]/al.sum(1)
                all_p.append(prob.cpu().numpy())
                all_l.append(np.concatenate([np.ones(npos), np.zeros(len(at)-npos)]))
        return compute_metrics(np.concatenate(all_p), np.concatenate(all_l))

    def load_embed(self):
        symbol_id={}; r2=json.load(open(os.path.join(self.ds, 'relation2ids')))
        e2=json.load(open(os.path.join(self.ds, 'ent2ids')))
        r2e=json.load(open(os.path.join(self.ds, 'relation2embids')))
        e2e=json.load(open(os.path.join(self.ds, 'ent2embids')))
        ee=np.load(os.path.join(self.ds, 'DRKG_TransE_entity.npy')); re=np.load(os.path.join(self.ds, 'DRKG_TransE_relation.npy'))
        i=0; emb=[]
        for k in sorted(r2):
            if k not in ['','OOV']: symbol_id[k]=i; i+=1; emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        for k in sorted(e2):
            if k not in ['','OOV']: symbol_id[k]=i; i+=1; emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        symbol_id['PAD']=i; emb.append(list(np.zeros((re.shape[1],))))
        self.symbol2id=symbol_id; self.symbol2vec=np.array(emb)


if __name__ == '__main__':
    args = read_options()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    os.makedirs('results', exist_ok=True)
    csv_path = 'results/ablation_curves.csv'
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['variant', 'iter', 'train_loss', 'dev_auroc', 'dev_f1', 'dev_acc'])

    trainer = EviDDIEAblationTrainer(dataset='dataset1')
    trainer.load_embed()

    for variant in ['softmax', 'w/o BSA', 'evi_no_evi', 'full_evi']:
        for layer in trainer.fc:
            if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        if variant == 'w/o BSA': trainer.linear_proj.reset_parameters()
        trainer.train_variant(variant, w, max_iter=MAX_ITER)

    f.close()
    logging.info(f'Done → {csv_path}')
