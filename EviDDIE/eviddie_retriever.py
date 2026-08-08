# zero_shot_retriever.py
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import FingerprintSimilarity


class ZeroShotDDIRetriever:
    """
    ExDDI-RV 检索方法的实现
    在零样本场景下，基于分子指纹相似度检索最相似的已知药物对
    """

    def __init__(self, train_data, embed_dim=128):
        self.train_data = train_data  # 已知DDI数据
        self.embed_dim = embed_dim
        self._build_fingerprint_cache()

    def _build_fingerprint_cache(self):
        """构建训练集中所有药物的指纹缓存"""
        self.fp_cache = {}
        self.ddi_cache = {}  # {(drug1_id, drug2_id): (label, explanation)}

        for task_name, data in self.train_data.items():
            for triple in data['triples']:
                drug1_smiles, drug2_smiles = triple[0], triple[2]

                # 计算MACCS指纹
                fp1 = self.smiles_to_fingerprint(drug1_smiles)
                fp2 = self.smiles_to_fingerprint(drug2_smiles)

                if drug1_smiles not in self.fp_cache:
                    self.fp_cache[drug1_smiles] = fp1
                if drug2_smiles not in self.fp_cache:
                    self.fp_cache[drug2_smiles] = fp2

                # 存储DDI信息
                key = (drug1_smiles, drug2_smiles)
                self.ddi_cache[key] = (1, triple[3] if len(triple) > 3 else "")

    @staticmethod
    def smiles_to_fingerprint(smiles):
        """将SMILES转换为MACCS指纹"""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return fp
        return None

    def tanimoto_similarity(self, fp1, fp2):
        """计算Tanimoto系数"""
        if fp1 is None or fp2 is None:
            return 0.0
        return FingerprintSimilarity(fp1, fp2)

    def retrieve_similar_ddi(self, query_drug1_smiles, query_drug2_smiles, k=50):
        """
        检索最相似的已知DDI对
        返回: (预测标签, 解释文本, 相似度分数)
        """
        query_fp1 = self.smiles_to_fingerprint(query_drug1_smiles)
        query_fp2 = self.smiles_to_fingerprint(query_drug2_smiles)

        if query_fp1 is None or query_fp2 is None:
            return 0, "Invalid SMILES", 0.0

        best_similarity = 0.0
        best_match = None
        best_label = 0
        best_explanation = ""

        # 为每个查询药物找到K个最相似的已知药物
        sim_scores1 = []
        sim_scores2 = []

        for train_smiles, train_fp in self.fp_cache.items():
            sim1 = self.tanimoto_similarity(query_fp1, train_fp)
            sim2 = self.tanimoto_similarity(query_fp2, train_fp)
            sim_scores1.append((train_smiles, sim1))
            sim_scores2.append((train_smiles, sim2))

        # 取top-K
        sim_scores1.sort(key=lambda x: x[1], reverse=True)
        sim_scores2.sort(key=lambda x: x[1], reverse=True)

        top_k1 = sim_scores1[:k]
        top_k2 = sim_scores2[:k]

        # 生成候选药物对
        candidate_pairs = []
        for d1, score1 in top_k1:
            for d2, score2 in top_k2:
                if d1 == d2:
                    continue
                pair_key = (d1, d2) if d1 < d2 else (d2, d1)
                pair_score = score1 * score2
                candidate_pairs.append((pair_key, pair_score))

        # 过滤并排序
        valid_candidates = []
        for pair_key, score in candidate_pairs:
            if pair_key in self.ddi_cache:
                valid_candidates.append((pair_key, score))

        if not valid_candidates:
            return 0, "No similar DDI found", 0.0

        # 取相似度最高的
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        best_pair_key, best_score = valid_candidates[0]

        best_label, best_explanation = self.ddi_cache[best_pair_key]

        return best_label, best_explanation, best_score

    def get_retrieval_embedding(self, query_drug1_smiles, query_drug2_smiles):
        """
        生成检索特征向量，用于融合到元学习模型
        返回: [similarity_score, has_match, explanation_length_norm]
        """
        label, explanation, score = self.retrieve_similar_ddi(
            query_drug1_smiles, query_drug2_smiles
        )

        # 构建特征向量
        features = torch.zeros(3)
        features[0] = score  # 相似度分数
        features[1] = 1.0 if label > 0 else 0.0  # 是否有匹配
        features[2] = min(len(explanation) / 100.0, 1.0)  # 解释长度归一化

        return features, explanation