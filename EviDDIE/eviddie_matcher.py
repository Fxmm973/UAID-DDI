#!/usr/bin/env Python
# coding=utf-8

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from eviddie_modules import *
from torch.autograd import Variable
import copy, math
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree
from eviddie_models import MVN_DDI


# (P0-7 cleanup: SynergisticGating and other MoseDTI remnants removed)


class Generate_Model(torch.nn.Module):
    '''
    Generator.
    '''

    def __init__(self, in_dim=700):  # BioSentVec output dimension
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_dim, out_features=256),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=256, out_features=512),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=512, out_features=64),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Distinguish_Model(torch.nn.Module):
    '''
    Discriminator.
    '''

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def rmse(predictions, targets):
    squared_error = torch.mean((predictions - targets) ** 2)
    rmse_val = pow(squared_error, 0.5)
    return rmse_val


def freeze_net(net):
    if not net:
        return

    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return

    for p in net.parameters():
        p.requires_grad = True


def directreturn(x):
    return x


class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        if p == 0:
            self.dropout = directreturn
        else:
            self.dropout = nn.Dropout(p)

    def forward(self, input):
        return self.dropout(input)


class GlobalAttentionPool(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx


class RelationRepresentation(nn.Module):
    def __init__(self, emb_dim, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(RelationRepresentation, self).__init__()
        self.RelationEncoder = TransformerEncoder(model_dim=emb_dim, ffn_dim=emb_dim * num_transformer_heads * 2,
                                                  num_heads=num_transformer_heads, dropout=dropout_rate,
                                                  num_layers=num_transformer_layers, max_seq_len=3,
                                                  with_pos=True)

    def forward(self, support, query=None):
        """
        forward
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return: [batch, dim]
        """

        relation = self.RelationEncoder(support, query)
        return relation


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim=100, ffn_dim=800, num_heads=4, dropout=0.1, num_layers=6, max_seq_len=3,
                 with_pos=True):
        super(TransformerEncoder, self).__init__()
        self.with_pos = with_pos
        self.num_heads = num_heads

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer2(model_dim * num_heads, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.rel_embed = nn.Parameter(torch.rand(1, model_dim), requires_grad=True)

    def repeat_dim(self, emb):
        """
        :param emb: [batch, t, dim]
        :return:
        """
        return emb.repeat(1, 1, self.num_heads)

    def forward(self, support, query=None):
        """
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return:
        """
        batch_size = 1
        if query == None:
            seq = torch.cat((self.rel_embed, support), dim=0)
        else:
            seq = torch.cat((support, query), dim=0)
        seq = seq.unsqueeze(0)
        pos = self.pos_embedding(batch_len=batch_size, seq_len=seq.shape[1])
        if self.with_pos:
            output = seq + pos
        else:
            output = seq
        output = self.repeat_dim(output)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
        if query == None:
            return output[:, 0, :]
        else:
            return output[:, 1:, :]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, batch_len, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([[1] + [2] * (seq_len - 1) for _ in range(batch_len)])
        return self.position_encoding(input_pos)


class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ScaledDotProductAttention2(nn.Module):
    """ Scaled Dot-Product Attention
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention2, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param attn_mask: [batch, time]
        :param scale:
        :param q: [batch, time, dim]
        :param k: [batch, time, dim]
        :param v: [batch, time, dim]
        :return:
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attn = attn * scale
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention2(nn.Module):
    """ Implement without batch dim"""

    def __init__(self, model_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention2, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention2(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        To be efficient, multi- attention is cal-ed in a matrix totally
        :param attn_mask:
        :param query: [batch, time, per_dim * num_heads]
        :param key:
        :param value:
        :return: [b, t, d*h]
        """
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attn = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attn


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.gelu = GELU()

    def forward(self, x):
        """

        :param x: [b, t, d*h]
        :return:
        """
        output = x.transpose(1, 2)
        output = self.w2(self.gelu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        output = self.layer_norm(x + output)
        return output


class EncoderLayer2(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        # ffn_dim
        super(EncoderLayer2, self).__init__()
        self.attention = MultiHeadAttention2(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


class GmpnnBlock(nn.Module):
    def __init__(self, edge_feats, n_feats, n_iter, dropout):
        super().__init__()
        self.n_feats = n_feats
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_n_feats = n_feats * 2

        self.w_i = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.w_j = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.a = nn.Parameter(torch.Tensor(1, self.n_feats))
        self.bias = nn.Parameter(torch.zeros(self.n_feats))

        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feats, self.n_feats)
        )

        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )

        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )

        glorot(self.w_i)
        glorot(self.w_j)
        glorot(self.a)

        self.sml_mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats)
        )

    def forward(self, data):
        edge_index = data.edge_index
        edge_feats = data.edge_feats
        edge_feats = self.edge_emb(edge_feats)

        deg = degree(edge_index[1], data.x.size(0), dtype=data.x.dtype)

        assert len(edge_index[0]) == len(edge_feats)
        alpha_i = (data.x @ self.w_i)
        alpha_j = (data.x @ self.w_j)
        alpha = alpha_i[edge_index[1]] + alpha_j[edge_index[0]] + self.bias
        alpha = self.sml_mlp(alpha)

        assert alpha.shape == edge_feats.shape
        alpha = (alpha * edge_feats).sum(-1)

        alpha = alpha / (deg[edge_index[0]])
        edge_weights = torch.sigmoid(alpha)

        assert len(edge_weights) == len(edge_index[0])
        edge_attr = data.x[edge_index[0]] * edge_weights.unsqueeze(-1)
        assert len(alpha) == len(edge_attr)

        out = edge_attr
        for _ in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]], data.line_graph_edge_index[1], dim_size=edge_attr.size(0),
                          dim=0, reduce='add')
            out = edge_attr + (out * edge_weights.unsqueeze(-1))

        x = data.x + scatter(out, edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.mlp(x)

        return x

    def mlp(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2

        return x


class AttentionSelectContext(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super(AttentionSelectContext, self).__init__()
        self.Bilinear = nn.Bilinear(dim, dim, 1, bias=False)
        self.Linear_tail = nn.Linear(dim, dim, bias=False)
        self.Linear_head = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def intra_attention(self, head, rel, tail, mask):
        """

        :param head: [b, dim]
        :param rel: [b, max, dim]
        :param tail:
        :param mask:
        :return:
        """
        head = head.unsqueeze(1).repeat(1, rel.size(1), 1)
        score = self.Bilinear(head, rel).squeeze(2)

        score = score.masked_fill_(mask, -np.inf)
        att = torch.softmax(score, dim=1).unsqueeze(dim=1)

        head = torch.bmm(att, tail).squeeze(1)
        return head

    def forward(self, left, right, mask_left=None, mask_right=None):
        """
        :param left: (head, rel, tail)
        :param right:
        :param mask_right:
        :param mask_left:
        :return:
        """
        head_left, rel_left, tail_left = left
        head_right, rel_right, tail_right = right

        weak_rel = head_right - head_left

        left = self.intra_attention(weak_rel, rel_left, tail_left, mask_left)
        right = self.intra_attention(weak_rel, rel_right, tail_right, mask_right)

        left = torch.relu(self.Linear_tail(left) + self.Linear_head(head_left))
        right = torch.relu(self.Linear_tail(right) + self.Linear_head(head_right))

        left = self.dropout(left)
        right = self.dropout(right)

        left = self.layer_norm(left + head_left)
        right = self.layer_norm(right + head_right)
        return left, right


class SRAE(nn.Module):
    """Stochastic Reconstruction-regularized Autoencoder (SRAE).

    Progressive bottleneck encoder (d -> d/2 -> d/4 -> d/8 -> d/2 -> d/4)
    with asymmetric noise injection: Gaussian perturbation (eta=1e-2) for
    support latents during training and near-deterministic encoding
    (eta=1e-3) for query/evaluation latents. No KL term and no
    standard-normal prior; the scale output is a learned noise magnitude."""

    def __init__(self, emb_dim):
        super(SRAE, self).__init__()

        self.conv_1 = nn.Linear(in_features=emb_dim, out_features=int(emb_dim / 2))
        self.conv_2 = nn.Linear(in_features=int(emb_dim / 2), out_features=int(emb_dim / 4))
        self.conv_3 = nn.Linear(in_features=int(emb_dim / 4), out_features=int(emb_dim / 8))

        self.fc_0 = nn.Linear(in_features=int(emb_dim / 8), out_features=int(emb_dim / 2))
        self.fc_1 = nn.Linear(in_features=int(emb_dim / 2), out_features=int(emb_dim / 4))
        self.fc_2 = nn.Linear(in_features=int(emb_dim / 2), out_features=int(emb_dim / 4))
        self.fc_3 = nn.Linear(in_features=int(emb_dim / 4), out_features=int(emb_dim / 4))

        self.fc_4 = nn.Linear(in_features=int(emb_dim / 4), out_features=emb_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        """Shared trunk: d -> d/2 -> d/4 -> d/8 (ReLU) -> d/2 (SeLU),
        then two heads producing the posterior mean and log-scale."""
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = F.selu(self.fc_0(x))
        return self.fc_1(x), self.fc_2(x)

    def sampling(self, z_mean, z_logvar, is_support, is_eval):
        """Asymmetric noise: eta=1e-2 Gaussian for support latents during
        training; eta=1e-3 (effectively deterministic) for query/eval."""
        std = torch.exp(0.5 * z_logvar)
        if is_eval:
            epsilon = 1e-3 * torch.ones_like(input=std)
        else:
            if is_support:
                epsilon = 1e-2 * torch.randn_like(input=std)
            else:
                epsilon = 1e-3 * torch.ones_like(input=std)
        return z_mean + std * epsilon

    def decode(self, z):
        """Two-layer decoder mapping the compact latent code back to d."""
        z = F.selu(self.fc_3(z))
        y_out = self.fc_4(z)
        y = y_out
        return y

    def forward(self, x, is_support=True, is_eval=False):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar, is_support, is_eval)
        y = self.decode(z)
        return y, z_mean, z_logvar, z


# Temporary checkpoint-compatibility alias for legacy checkpoints and imports.
VAE = SRAE


class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings
    """

    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.2, batch_size=64,
                 finetune=False, aggregate='max', task_emb=None):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim

        self.pad_idx = num_symbols
        self.dropout = nn.Dropout(0.5)

        d_model = self.embed_dim * 2

        in_feats = 70
        hid_feats = 64
        dropout = 0.5
        n_iter = 3
        edge_feats = 6
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.BatchNorm1d(hid_feats),
            CustomDropout(dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats),
            CustomDropout(dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats),
            CustomDropout(dropout),
            nn.PReLU(),
        )
        self.h_gpool = GlobalAttentionPool(2 * hid_feats)
        self.t_gpool = GlobalAttentionPool(2 * hid_feats)
        self.w_j = nn.Linear(2 * hid_feats, 2 * hid_feats)
        self.w_i = nn.Linear(2 * hid_feats, 2 * hid_feats)
        self.prj_j = nn.Linear(2 * hid_feats, 2 * hid_feats)
        self.prj_i = nn.Linear(2 * hid_feats, 2 * hid_feats)
        self.lin = nn.Sequential(
            nn.Linear(hid_feats * 2, hid_feats * 2),
            nn.PReLU(),
            nn.Linear(hid_feats * 2, d_model),
        )
        self.propagation_layer = GmpnnBlock(edge_feats, hid_feats, n_iter, dropout)
        self.i_pro = nn.Parameter(torch.zeros(hid_feats * 2, hid_feats))
        self.j_pro = nn.Parameter(torch.zeros(hid_feats * 2, hid_feats))
        glorot(self.i_pro)
        glorot(self.j_pro)

        n_atom_feats = 55
        kge_dim = 128
        rel_total = 0
        self.model = MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, [64, 64],
                             [2, 2], 64, 0.0)
        # Dual-output evidential head: raw evidence scores for the two classes;
        # Softplus is applied in forward() to guarantee non-negative evidence.
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            CustomDropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # dual-output evidential head
        )

        self.pad_idx = num_symbols
        self.fc_struc_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            CustomDropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 128)
        )

        self.num_transformer_layers = 2
        self.num_transformer_heads = 1
        self.dropout_layers = 0.2
        self.RelationRepresentation = RelationRepresentation(emb_dim=embed_dim * 2,
                                                             num_transformer_layers=self.num_transformer_layers,
                                                             num_transformer_heads=self.num_transformer_heads,
                                                             dropout_rate=self.dropout_layers)
        self.RelationRepresentation2 = RelationRepresentation(emb_dim=embed_dim * 2,
                                                              num_transformer_layers=self.num_transformer_layers,
                                                              num_transformer_heads=self.num_transformer_heads,
                                                              dropout_rate=self.dropout_layers)

        self.vaemodel = SRAE(emb_dim=embed_dim * 2)  # attribute name kept for checkpoint compatibility

    def structure_encoder(self, batch):
        qdrug_data_origin, qunique_drug_pair, qrels, qdrug_pair_indices, qnode_j_for_pairs, qnode_i_for_pairs, qdrug_pair_list, qdrug_node_num_pair_list = batch
        drug_data = copy.deepcopy(qdrug_data_origin)
        drug_data.x = self.mlp(drug_data.x)
        new_feats = self.propagation_layer(drug_data)
        drug_data.x = new_feats
        x_j = drug_data.x[qnode_j_for_pairs]
        x_i = drug_data.x[qnode_i_for_pairs]
        g_new_feats = self.h_gpool(drug_data.x, drug_data.edge_index, drug_data.batch)
        g_h_align = g_new_feats[qdrug_pair_list[:, 0]].repeat_interleave(qdrug_node_num_pair_list[:, 1], dim=0)
        g_t_align = g_new_feats[qdrug_pair_list[:, 1]].repeat_interleave(qdrug_node_num_pair_list[:, 0], dim=0)
        h_scores = (self.w_j(x_j) * self.prj_i(g_t_align)).sum(-1)
        h_scores = softmax(h_scores, qunique_drug_pair['j_indices'], dim=0)
        t_scores = (self.w_i(x_i) * self.prj_j(g_h_align)).sum(-1)
        t_scores = softmax(t_scores, qunique_drug_pair['i_indices'], dim=0)
        x_j = x_j * g_t_align * h_scores.unsqueeze(-1)
        x_i = x_i * g_h_align * t_scores.unsqueeze(-1)
        query_pair = self.lin(torch.cat([scatter(x_i[qunique_drug_pair.edge_index[1]] @ self.i_pro,
                                                 qunique_drug_pair.edge_index_batch, reduce='add', dim=0)[
                                             qdrug_pair_indices],
                                         scatter(x_j[qunique_drug_pair.edge_index[0]] @ self.j_pro,
                                                 qunique_drug_pair.edge_index_batch, reduce='add', dim=0)[
                                             qdrug_pair_indices]], dim=-1))
        return query_pair

    def forward(self, task_proto, query, support, query_meta=None, support_meta=None, query_batch=None,
                support_batch=None, optim_VAE=None, is_eval=False, trainGAN=False):
        '''
        EviDDIE forward (canonical): molecular CSE features -> pair
        concatenation -> SRAE latent -> dual-output evidential comparator.
        '''

        if trainGAN == True:
            # GAN step: support-set CSE features only
            support_left_, support_right_ = self.model(support_batch)
            support_neighbor = torch.cat((support_left_, support_right_), dim=-1)

            output_s, z_mean_s, z_logvar_s, zs = self.vaemodel(support_neighbor, is_support=True, is_eval=is_eval)
            return zs

        else:
            # Query branch: CSE features only (EviDDIE uses no KG neighbor
            # aggregation or gating, by design)
            query_left_, query_right_ = self.model(query_batch)
            query_neighbor = torch.cat((query_left_, query_right_), dim=-1)

            if is_eval == False:
                support_left_, support_right_ = self.model(support_batch)
                support_neighbor = torch.cat((support_left_, support_right_), dim=-1)
                support = support_neighbor

            query = query_neighbor

            if is_eval == False:
                output_s, z_mean_s, z_logvar_s, zs = self.vaemodel(support, is_support=True, is_eval=is_eval)

            output_q, z_mean_q, z_logvar_q, zq = self.vaemodel(query, is_support=False, is_eval=is_eval)

            evidence = F.softplus(self.fc(torch.abs(task_proto.expand_as(zq) - zq)))
            alpha = evidence + 1

            if is_eval == False:
                ls = F.mse_loss(input=output_s, target=support.detach(), reduction='mean')
                lq = F.mse_loss(input=output_q, target=query.detach(), reduction='mean')
                mse_loss = (ls + lq) / 2
                return alpha, 0.2 * mse_loss
            else:
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                return prob[:, 1], 0

    def forward_(self, query_meta, support_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1)  # tanh
        support = torch.cat((support_left, support_right), dim=-1)  # tanh

        support_expand = support.expand_as(query)

        distances = F.sigmoid(self.siamese(torch.abs(support_expand - query))).squeeze()
        return distances


class RescalMatcher(nn.Module):
    """
    Matching based on KB Embeddings
    """

    def __init__(self, embed_dim, num_ents, num_rels, use_pretrain=True, ent_embed=None, rel_matrices=None, dropout=0.1,
                 attn_layers=1, n_head=4, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(RescalMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.ent_emb = nn.Embedding(num_ents + 1, embed_dim, padding_idx=num_ents)
        self.rel_matrices = nn.Embedding(num_rels + 1, embed_dim * embed_dim, padding_idx=num_rels)

        self.aggregate = aggregate

        self.gcn_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.dropout = nn.Dropout(0.5)

        init.xavier_normal(self.gcn_w.weight)
        init.constant(self.gcn_b, 0)

        if use_pretrain:
            print('LOADING KB EMBEDDINGS')
            self.ent_emb.weight.data.copy_(torch.from_numpy(ent_embed))
            self.rel_matrices.weight.data.copy_(torch.from_numpy(rel_matrices))
            if not finetune:
                print('FIX KB EMBEDDING')
                self.ent_emb.weight.requires_grad = False
                self.rel_matrices.weight.requires_grad = False

        d_model = embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.rel_matrices(relations))
        ent_embeds = self.dropout(self.ent_emb(entities))

        batch_size = rel_embeds.size()[0]
        max_neighbors = rel_embeds.size()[1]
        rel_embeds = rel_embeds.view(-1, self.embed_dim, self.embed_dim)
        ent_embeds = ent_embeds.view(-1, self.embed_dim).unsqueeze(2)

        concat_embeds = torch.bmm(rel_embeds, ent_embeds).squeeze().view(batch_size, max_neighbors, -1)

        out = self.gcn_w(concat_embeds) + self.gcn_b
        out = torch.sum(out, dim=1)
        out = out / num_neighbors
        out = F.tanh(out)
        return out

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        if query_meta == None:
            support = self.dropout(self.symbol_emb(support)).view(-1, 2 * self.embed_dim)
            query = self.dropout(self.symbol_emb(query)).view(-1, 2 * self.embed_dim)
            support = support.unsqueeze(0)
            support_g = self.support_encoder(support).squeeze(0)
            query_f = self.query_encoder(support_g, query)
            matching_scores = torch.matmul(query_f, support_g.t())
            if self.aggregate == 'max':
                query_scores = torch.max(matching_scores, dim=1)[0]
            elif self.aggregate == 'mean':
                query_scores = torch.mean(matching_scores, dim=1)
            elif self.aggregate == 'sum':
                query_scores = torch.sum(matching_scores, dim=1)
            return query_scores

        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat((support_left, support_right), dim=-1)  # tanh

        support = support_neighbor
        query = query_neighbor
        support_g = self.support_encoder(support)
        query_g = self.support_encoder(query)
        query_f = self.query_encoder(support_g, query_g)
        matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

        return matching_scores

    def forward_(self, query_meta, support_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1)
        support = torch.cat((support_left, support_right), dim=-1)

        support_expand = support.expand_as(query)

        distances = F.sigmoid(self.siamese(torch.abs(support_expand - query))).squeeze()
        return distances