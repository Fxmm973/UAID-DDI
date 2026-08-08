import math
import datetime
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):      

        rels = self.rel_emb(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)
        

        if alpha_scores is not None:
          scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
       
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim,dp,head,edge,head_out_feats):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim,head_out_feats//2,head,edge_dim=edge,dropout=dp)
    
    def forward(self,data):
        input_feature,edge_index = data.x, data.edge_index
        input_feature = F.relu(input_feature)
        intra_rep = self.intra(input_feature,edge_index,data.edge_attr)
        return intra_rep

class InterGraphAttention(nn.Module):
    def __init__(self, input_dim,dp,head,edge,head_out_feats):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim,input_dim),head_out_feats//2,head,dropout=dp)
    
    def forward(self,h_data,t_data,b_graph):
        edge_index = b_graph.edge_index
        h_input = F.relu(h_data.x)
        t_input = F.relu(t_data.x)
        t_rep = self.inter((h_input,t_input),edge_index)
        h_rep = self.inter((t_input,h_input),edge_index[[1,0]])
        return h_rep,t_rep

class MergeFD(nn.Module):
    def __init__(self, in_features_fp, in_features_desc, kge_dim):
        super().__init__()
        self.in_features_fp = in_features_fp
        self.in_features_desc = in_features_desc
        self.kge_dim = kge_dim
        self.reduction_fp = nn.Sequential(nn.Linear(self.in_features_fp, 512),
                                          nn.ELU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(512, self.kge_dim),
                                          nn.ELU(),
                                          nn.Dropout(0.3)
                                          )
        self.reduction_desc = nn.Sequential(nn.Linear(self.in_features_desc, 256),
                                          nn.ELU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(256, self.kge_dim),
                                          nn.ELU(),
                                          nn.Dropout(0.3))
        self.merge_fd = nn.Sequential(nn.Linear(self.kge_dim, self.kge_dim),
                                   nn.ELU())
        
    def forward(self,h_data_fin,h_data_desc,t_data_fin,t_data_desc):
        h_data_fin = F.normalize(h_data_fin, 2, 1)
        t_data_fin = F.normalize(t_data_fin, 2 ,1)
        h_data_fin = self.reduction_fp(h_data_fin)
        t_data_fin = self.reduction_fp(t_data_fin)
        h_fdmerge = h_data_fin
        h_fdmerge = F.normalize(h_fdmerge, 2, 1)
        h_fdmerge = self.merge_fd(h_fdmerge)
        t_fdmerge = t_data_fin
        t_fdmerge = F.normalize(t_fdmerge, 2, 1)
        t_fdmerge = self.merge_fd(t_fdmerge)

        return h_fdmerge, t_fdmerge, h_data_fin, h_data_desc, t_data_fin, t_data_desc




# ================= 药效团感知TransformerConv =================
import torch
from torch import nn
from torch_geometric.nn import TransformerConv

class PharmacophoreAwareTransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, edge_dim, dropout):
        super().__init__()
        self.base_conv = TransformerConv(in_channels, out_channels, heads,
                                             edge_dim=edge_dim, dropout=dropout)
        self.pharmacophore_types = ['H-bond-donor', 'H-bond-acceptor',
                                    'hydrophobic', 'aromatic', 'charged']#映射 5 类经典药效团
        self.pharm_weight = nn.Parameter(torch.ones(len(self.pharmacophore_types)))
        self.gate_fc = nn.Linear(in_channels, 1)

    def detect_pharmacophore(self, x):
        score = torch.zeros(x.size(0), len(self.pharmacophore_types), device=x.device)
        # Element-based pharmacophore proxies (element-level heuristics)
        score[:, 0] = x[:, 1]              # N atom as H-bond donor proxy
        score[:, 1] = x[:, 2]              # O atom as H-bond acceptor proxy
        score[:, 2] = x[:, 0]              # C atom as hydrophobic proxy
        score[:, 3] = x[:, 53] if x.size(1) > 53 else 0.  # RDKit aromatic flag
        score[:, 4] = torch.abs(x[:, 46]) if x.size(1) > 46 else 0.  # formal charge
        return score

    def forward(self, x, edge_index, edge_attr):
        pharm_score = self.detect_pharmacophore(x)
        pharm_boost = torch.sigmoid(
            (pharm_score * self.pharm_weight.unsqueeze(0)).sum(1, keepdim=True))
        gate = torch.sigmoid(self.gate_fc(x))
        adaptive_weight = 0.7 * gate + 0.3 * pharm_boost#70%来自原始特征，30%来自药效团（7:3）
        x_conv = self.base_conv(x, edge_index, edge_attr)
        return x_conv * (1 + 1.5 * adaptive_weight)#具有重要药效团特征的节点在信息传递中获得更大权重
    # ===========================================================
 