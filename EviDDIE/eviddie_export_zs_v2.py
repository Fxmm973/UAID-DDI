#!/usr/bin/env Python
# coding=utf-8
"""
导出 EviDDIE zero-shot 消融变体预测（v2：使用固定负样本 manifest）
改动：
- 从预生成的 manifest JSON 读取负样本，不再动态采样
- SEEDS 扩展到 5 个
- 负样本固定后跨方法一致
"""
import sys
import os
import hashlib

# 允许从任意目录启动：把仓库根目录加入 sys.path（shared/ 位于仓库根）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

# 旧模块名 → 新模块名映射（兼容旧 checkpoint 的 pickle 路径）
import eviddie_matcher
import eviddie_modules
import eviddie_models
import eviddie_layers
_OLD_TO_NEW = {
    'matcher_structure_acc_fp_neigh_VAE_GAN_struc': eviddie_matcher,
    'matcher_structure_acc_fp_neigh_VAE_GAN_struc_ttt': eviddie_matcher,
    'modules_structure_fp_neigh': eviddie_modules,
    'models_t_struc': eviddie_models,
    'models_t_struc_ttt': eviddie_models,
    'layers': eviddie_layers,
}
for _old, _new in _OLD_TO_NEW.items():
    sys.modules[_old] = _new

import torch.nn.functional as F
import csv
from torch.autograd import Variable
from eviddie_args import read_options
from eviddie_dataloader import *
from eviddie_matcher import EmbedMatcher
from shared.checkpoint import load_state_dict_safe

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

METHOD_MAP = {'softmax': 'Softmax baseline', 'evi_no_evi': 'EviDDIE w/o EVI', 'full_evi': 'EviDDIE'}
SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]  # 扩展到5个
# fc 输出通道约定：0 = 负类 (negative)，1 = 正类 (positive)；prob = alpha[:,1]/S。
CLASS_ORDER = ('negative', 'positive')


def load_neg_manifest(dataset, split, seed):
    """Load pre-generated negative manifest."""
    path = f'neg_manifests/{split}_seed{seed}_negatives.json'
    with open(path) as f:
        return json.load(f)


class ExportVariants(object):
    def __init__(self, arg):
        for k, v in vars(arg).items(): setattr(self, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device}")

        self.semantic_task = json.load(open(f'{arg.dataset}/{arg.semantic}'))
        # P0-7: inference uses the raw BioSentVec embeddings (no semantic noise).
        # P0-2 审计：显式形状检查（700 维）+ 固定 key 排序，防止原型错位。
        ordered_keys = sorted(list(self.semantic_task.keys()))
        for task in ordered_keys:
            vector = np.asarray(self.semantic_task[task], dtype=np.float32).reshape(-1)
            if vector.shape != (700,):
                raise ValueError(f'{task}: expected 700-d BioSentVec vector, got {vector.shape}')
            self.semantic_task[task] = vector
        self.task_ebmedding = []
        self.task2id = {}
        for num,i in enumerate(ordered_keys):
            self.task2id[i]=num; self.task_ebmedding.append(self.semantic_task[i])
        self.task_ebmedding = torch.tensor(np.vstack(self.task_ebmedding)).float().to(self.device)

        self.use_pretrain = not self.random_embed
        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys())-1
        self.pad_id = self.num_symbols

        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols,
                                     use_pretrain=self.use_pretrain, embed=self.symbol2vec,
                                     dropout=self.dropout, batch_size=self.batch_size,
                                     finetune=self.fine_tune, aggregate=self.aggregate,
                                     task_emb=self.task_ebmedding).to(self.device)
        self.matcher.eval()

        ckpt = torch.load(arg.pretrained_model, map_location=self.device)
        # P0-3 (GPT 4.1)：禁止旧单输出检查点的 1->2 拼接转换——该转换未经 EDL 损失训练，
        # 不能产生可信的双类 Dirichlet evidence。旧检查点直接硬失败，提示重训。
        head_weight = ckpt.get('fc.5.weight')
        head_bias = ckpt.get('fc.5.bias')
        if head_weight is None or head_bias is None:
            raise RuntimeError('Checkpoint does not contain the evidential head (fc.5.*)')
        if head_weight.shape[0] != 2 or head_bias.shape[0] != 2:
            raise RuntimeError(
                'Legacy single-output checkpoint is unsupported. '
                'Retrain EviDDIE with the native two-class evidential head.')
        for k in list(ckpt.keys()):
            if any(x in k for x in ['symbol_emb','gcn_w','gcn_b','Bilinear','Linear_self',
                                     'Linear_nei','Linear_weak_rel','NeighborAggregator','siamese',
                                     'support_encoder','query_encoder']):
                del ckpt[k]
        load_state_dict_safe(self.matcher, ckpt, model_name='matcher')

        self.G_m = torch.load(arg.g_model_path, map_location=self.device)
        self.G_m.eval()

        self.ent2id = json.load(open(self.dataset+'/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.rel2candidates = json.load(open(self.dataset+'/rel2candidates.json'))
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset+'/e1rel_e2.json'))

        # ---- 加载负样本 manifest（关键改动）----
        # P0-5 (6.1.3)：收集真实评估 episode，导出结束后存档
        self.episode_manifest = {}
        self.neg_manifests = {}
        for split in ['dev', 'test', 'test2']:
            self.neg_manifests[split] = {}
            for seed in SEEDS:
                self.neg_manifests[split][seed] = load_neg_manifest(arg.dataset, split, seed)

    def load_embed(self):
        symbol_id={}; rel2id=json.load(open(self.dataset+'/relation2ids'))
        ent2id=json.load(open(self.dataset+'/ent2ids'))
        r2e=json.load(open(self.dataset+'/relation2embids'))
        e2e=json.load(open(self.dataset+'/ent2embids'))
        ee=np.load(self.dataset+'/DRKG_TransE_entity.npy')
        re=np.load(self.dataset+'/DRKG_TransE_relation.npy')
        i=0; emb=[]
        for k in rel2id:
            if k not in ['','OOV']: symbol_id[k]=i; i+=1; emb.append(list(re[r2e[k],:]) if r2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        for k in ent2id:
            if k not in ['','OOV']: symbol_id[k]=i; i+=1; emb.append(list(ee[e2e[k],:]) if e2e[k]!=-1 else list(np.random.randn(re.shape[1])))
        symbol_id['PAD']=i; emb.append(list(np.zeros((re.shape[1],))))
        self.symbol2id=symbol_id; self.symbol2vec=np.array(emb)

    def build_connection(self, max_=100):
        self.connections=(np.ones((self.num_ents,max_,2))*self.pad_id).astype(int)
        self.e1_rele2=defaultdict(list); self.e1_degrees=defaultdict(int)
        with open(self.dataset+'/path_graph') as f:
            for line in tqdm(f.readlines(),desc='Connections'):
                e1,rel,e2=line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel],self.symbol2id[e2]))
        for ent,id_ in self.ent2id.items():
            nb=self.e1_rele2[ent]
            if len(nb)>max_: random.shuffle(nb); nb=nb[:max_]
            self.e1_degrees[id_]=len(nb)
            for idx,_ in enumerate(nb): self.connections[id_,idx,0]=_[0]; self.connections[id_,idx,1]=_[1]

    def get_meta(self, left, right):
        lc=Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left],axis=0))).to(self.device)
        ld=Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        rc=Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right],axis=0))).to(self.device)
        rd=Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).to(self.device)
        return (lc,ld,rc,rd)

    def load_head(self, variant):
        path = f'{self.save_dir}/fc_{variant}.pt'
        head = torch.load(path, map_location=self.device)
        self.matcher.fc.load_state_dict(head)
        self.matcher.eval()
        w = next(iter(head.values()))
        logging.info(f'Loaded fc head: {variant} (first weight mean_abs={w.float().abs().mean().item():.5f})')

    def export(self, mode, csv_writer, train_seed, eval_seed, method_name, variant):
        setting_map = {'dev':'common','test':'fewer','test2':'rare'}
        setting = setting_map.get(mode, mode)
        logging.info(f'[{method_name}] {mode.upper()} (train_seed={train_seed})')

        if mode=='dev': test_tasks=json.load(open(self.dataset+'/dev_tasks.json'))
        elif mode=='test': test_tasks=json.load(open(self.dataset+'/test_tasks.json'))
        else: test_tasks=json.load(open(self.dataset+'/test2_tasks.json'))
        rel2id=json.load(open(self.dataset+'/relation2ids'))

        # 读取预生成的固定负样本
        neg_manifest = self.neg_manifests[mode][eval_seed]

        with torch.no_grad():
            for query_ in test_tasks.keys():
                query_triples = test_tasks[query_][0:]  # few=0 for zero-shot
                if not query_triples: continue

                # 从固定 manifest 读取负样本（不再动态采样）
                manifest_entries = neg_manifest.get(query_, [])
                false_triples = []
                for entry in manifest_entries:
                    d_i, d_j, d_k, rel = entry
                    false_triples.append([d_i, rel, d_k])

                # 确保数量匹配
                if len(false_triples) != len(query_triples):
                    logging.warning(f'{query_}: manifest has {len(false_triples)} negs but {len(query_triples)} queries, skipping')
                    continue

                # P0-5 (6.1.3)：记录真实评估 episode（零样本：无 support）
                self.episode_manifest[f'{mode}:{query_}'] = {
                    'query_positives': [list(x) for x in query_triples],
                    'query_negatives': [list(x) for x in false_triples],
                }

                all_triples = query_triples + false_triples
                all_rel2id = [[t[0],t[2],rel2id[t[1]]] for t in all_triples]
                q_left = [self.ent2id[t[0]] for t in all_triples]
                q_right = [self.ent2id[t[2]] for t in all_triples]
                q_meta = self.get_meta(q_left, q_right)
                n_pos = len(query_triples)

                qb = DrugDataset(all_rel2id)
                qbl = DrugDataLoader(qb, batch_size=len(all_rel2id), shuffle=False)
                qb_data = [t.to(self.device) for t in next(iter(qbl))]
                task_emb = self.G_m(self.task_ebmedding[self.task2id[query_]]).detach()

                ql_, qr_ = self.matcher.model(qb_data)
                qn = torch.cat((ql_, qr_), dim=-1)
                _, _, _, zq = self.matcher.vaemodel(qn, is_support=False, is_eval=True)
                fc_out = self.matcher.fc(torch.abs(task_emb.expand_as(zq) - zq))

                if variant == 'softmax':
                    probs = F.softmax(fc_out, dim=1)[:, 1]
                    unc = 1.0 - torch.max(F.softmax(fc_out, dim=1), dim=1)[0]
                else:
                    evidence = F.softplus(fc_out)
                    alpha = evidence + 1
                    prob = alpha / alpha.sum(dim=1, keepdim=True)
                    probs = prob[:, 1]
                    unc = 2.0 / alpha.sum(dim=1)

                probs_np = probs.cpu().numpy()
                unc_np = unc.cpu().numpy()
                gt = np.concatenate([np.ones(n_pos), np.zeros(len(all_triples)-n_pos)])

                # ---- P0-2 审计断言：类别顺序、标签拼接与概率范围 ----
                # fc 输出通道约定：0 = 负类 (negative)，1 = 正类 (positive)
                assert CLASS_ORDER == ('negative', 'positive'), 'class order convention changed'
                assert int(gt[:n_pos].sum()) == n_pos, 'positive labels must fill the first n_pos rows'
                assert int(gt[n_pos:].sum()) == 0, 'negative labels must fill the remaining rows'
                assert np.all((probs_np >= 0.0) & (probs_np <= 1.0)), 'probs out of [0,1]'

                rm = getattr(self, 'run_meta', {})
                for idx, (t, p, u) in enumerate(zip(all_triples, probs_np, unc_np)):
                    csv_writer.writerow([rm.get('run_id', ''), train_seed, eval_seed, setting, 0,
                                         method_name, query_, t[0], t[2], int(gt[idx]),
                                         1 if p >= 0.5 else 0,
                                         round(float(p), 8), round(float(u), 8),
                                         rm.get('checkpoint_sha256', ''),
                                         rm.get('eval_manifest_hashes', {}).get(mode, ''),
                                         rm.get('event_embedding_sha256', ''),
                                         rm.get('git_commit', '')])


if __name__ == '__main__':
    args = read_options()
    args.dataset = 'dataset1'
    args.no_meta = False
    args.random_embed = False
    args.save_dir = 'models/dataset1'

    TRAINING_SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]
    EVAL_MANIFEST_SEED = 19940419  # 固定负样本种子
    MODES = ['dev', 'test', 'test2']

    # ---- P0-3 / GPT 4.4：逐样本 CSV 元数据 ----
    def sha256_file(p):
        return hashlib.sha256(open(p, 'rb').read()).hexdigest()

    eval_manifest_hashes = {
        split: sha256_file(f'neg_manifests/{split}_seed{EVAL_MANIFEST_SEED}_negatives.json')
        for split in MODES
    }
    embedding_sha = sha256_file(f'{args.dataset}/event_embedding2.json')
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                             stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = 'unknown'
    logging.info(f'[META] eval manifest hashes: {eval_manifest_hashes}')
    logging.info(f'[META] event embedding sha256: {embedding_sha}')
    logging.info(f'[META] git commit: {git_commit}')

    out_dir = 'results/predictions'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'predictions_dataset1_zero_shot_variants.csv')

    import hashlib  # 种子独立性验证
    ckpt_hashes = []

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['run_id','train_seed','eval_seed','setting','shot','method','event_type','drug_a','drug_b',
                     'y_true','y_pred','prob','uncertainty','checkpoint_sha256','eval_manifest_sha256',
                     'event_embedding_sha256','git_commit'])

        for train_seed in TRAINING_SEEDS:
            args.pretrained_model = f'models/dataset1/eviddie_0shot_seed{train_seed}/bestmodel'
            args.g_model_path = f'models/dataset1/eviddie_0shot_seed{train_seed}/bestmodel_G'
            if not os.path.exists(args.pretrained_model):
                raise FileNotFoundError(
                    f'Per-seed checkpoint not found: {args.pretrained_model}. '
                    f'Per-seed exports must use the checkpoint of the corresponding training seed.')
            ckpt_hashes.append(hashlib.sha256(open(args.pretrained_model, 'rb').read()).hexdigest())
            eval_seed = EVAL_MANIFEST_SEED
            # P0-3 / GPT 4.4：每个 CSV 行携带完整元数据
            args.run_meta = {
                'run_id': f'eviddie-{train_seed}-{eval_seed}',
                'checkpoint_sha256': ckpt_hashes[-1],
                'eval_manifest_hashes': eval_manifest_hashes,
                'event_embedding_sha256': embedding_sha,
                'git_commit': git_commit,
            }

            random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(eval_seed)

            for variant in ['softmax', 'evi_no_evi', 'full_evi']:
                method = METHOD_MAP[variant]
                logging.info(f'===== train_seed={train_seed} {method} =====')
                ex = ExportVariants(args)
                if variant == 'full_evi':
                    # 正式模型：直接使用 per-seed checkpoint 自带的原生双输出 EDL 头，
                    # 绝不用共享的冻结骨干消融头覆盖（这是此前导出概率恒为 0.5 的原因）
                    logging.info('[HEAD] full_evi: keeping the checkpoint native EDL head.')
                else:
                    ex.load_head(variant)
                for mode in MODES:
                    ex.export(mode, w, train_seed, eval_seed, method, variant)
                # P0-5 (6.1.3)：保存真实评估 episode manifest（每个 (train_seed, variant) 一份）
                if variant == 'full_evi':
                    em_dir = 'results/predictions/episode_manifests'
                    os.makedirs(em_dir, exist_ok=True)
                    em_path = os.path.join(em_dir, f'episode_manifest_0shot_seed{train_seed}.json')
                    payload = {
                        'shot': 0,
                        'train_seed': train_seed,
                        'eval_manifest_seed': eval_seed,
                        'episodes': ex.episode_manifest,
                    }
                    with open(em_path, 'w', encoding='utf-8') as ef:
                        json.dump(payload, ef, ensure_ascii=False)
                    logging.info(f'[P0-5] Episode manifest saved: {em_path} '
                                 f'({len(ex.episode_manifest)} episodes)')

        # ---- 种子独立性验证：5 train_seed -> 5 不同 checkpoint 路径 -> 5 不同哈希 ----
        if len(set(ckpt_hashes)) != len(ckpt_hashes):
            raise RuntimeError(
                f'EviDDIE zero-shot: checkpoint hashes are not unique across training seeds: '
                f'{len(set(ckpt_hashes))} distinct of {len(ckpt_hashes)}')
        logging.info(f'[SEED-CHAIN] {len(set(ckpt_hashes))} distinct checkpoint hashes across '
                     f'{len(ckpt_hashes)} training seeds (OK).')
        logging.info(f'[SEED-CHAIN] Evaluation manifest seed fixed to {EVAL_MANIFEST_SEED} '
                     f'for all seeds (identical manifest hash across seeds).')

    logging.info(f'Done! Saved to {out_csv}')
    logging.info(f'Training seeds: {TRAINING_SEEDS}, Fixed eval manifest: {EVAL_MANIFEST_SEED}')
