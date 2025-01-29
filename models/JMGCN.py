# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from geopy.distance import geodesic
import numpy as np
from lib.utils import scaled_Laplacian, cheb_polynomial
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj
from util import get_L_d, compute_position_vectors, load_long_static_matrix
import os
import math
import pandas as pd


def cheb_polynomial_torch1(L_tilde, K):

    N = L_tilde.shape[-1]
    cheb_polynomials = [torch.eye(N).unsqueeze(0).repeat(L_tilde.shape[0], 1, 1).to(L_tilde.device), L_tilde.clone()]
    for i in range(2, K):
        next_cheb = 2 * torch.matmul(L_tilde, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2]
        cheb_polynomials.append(next_cheb)
    return cheb_polynomials


def cheb_polynomial_torch(L_tilde, K):

    N = L_tilde.shape[0]
    cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde.clone()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask, last_att):
        '''
        scores : [batch_size, n_heads, len_q, len_k]
        '''
        last_att = last_att
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + last_att
        if attn_mask is not None:
            attn_mask = attn_mask < 0.90  # mask
            scores.masked_fill_(attn_mask, -1e9)
        cycle = scores
        return cycle, scores


class ST_Attention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads):
        super(ST_Attention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask, last_att):
        '''
        input_Q: [batch_size, n_heads, len_q, d_k]
        input_K: [batch_size, n_heads, len_k, d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_heads, 1, 1)

        cycle, attn = ScaledDotProductAttention(self.d_k)(Q, K, attn_mask, last_att)
        return cycle, attn


class cheb_conv_kagcn(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, fusiongraph, in_channels, out_channels, num_of_vertices, position, distance, DEVICE):
        '''
        :param K: int
        :param in_channels: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_kagcn, self).__init__()

        self.K = K
        self.DEVICE = DEVICE  # 改动
        self.fusiongraph = fusiongraph
        self.position_vectors = position
        self.distance = distance
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.Theta_L = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

        adj_init = nn.Parameter(torch.randn(256, 34, 34))  # initial design
        adj_init_bias = nn.Parameter(torch.randn(34, 34))
        self.adj_init_bias = nn.Parameter(adj_init_bias.to(DEVICE), requires_grad=True)
        self.adj_init = nn.Parameter(adj_init.to(DEVICE), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.adj_init.size(1))
        self.adj_init.data.uniform_(-stdv, stdv)
        if self.adj_init_bias is not None:
            self.adj_init_bias.data.uniform_(-stdv, stdv)

    def forward(self, x, st_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        adj_for_run = self.fusiongraph()
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            # dynamic graph of wind
            adj_for_run_d = self.construct_dynamic_adjacency_matrix(graph_signal)
            adj_dy = F.normalize(adj_for_run_d, p=2, dim=(1, 2))

            # dynamic L_matrix
            l_d = get_L_d(adj_dy, self.DEVICE)
            cheb_polynomials_d1 = cheb_polynomial_torch1(l_d, self.K)

            L_d = get_L_d(adj_dy.transpose(1, 2), self.DEVICE)
            cheb_polynomials_d2 = cheb_polynomial_torch1(L_d, self.K)

            for k in range(self.K):

                T_k1 = cheb_polynomials_d1[k]  # (b, N, N)
                T_k2 = cheb_polynomials_d2[k]

                myspatial_attention = st_attention[:, k, :, :]
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                T_k_with_att1 = T_k1.mul(myspatial_attention)
                T_k_with_att2= T_k2.mul(myspatial_attention)

                theta_k1 = self.Theta[k]  # (in_channel, out_channel)
                theta_k2 = self.Theta_L[k]

                # spatial_attention graph convolution, including forward and backward.
                rhs = torch.matmul(T_k_with_att1.permute(0, 2, 1), graph_signal).matmul(theta_k1) + \
                         torch.matmul(T_k_with_att2.permute(0, 2, 1), graph_signal).matmul(theta_k2)
                output = output + rhs

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return self.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

    def construct_dynamic_adjacency_matrix(self, graph_signal):

        num_samples, num_vertices, num_features = graph_signal.shape
        adjacency_matrix = torch.zeros(num_samples, num_vertices, num_vertices).to(self.DEVICE)

        cosine_sims = F.cosine_similarity(graph_signal[:, :, 1:3].unsqueeze(2),
                                          self.position_vectors.unsqueeze(0).unsqueeze(0), dim=-1)

        wind_speed = torch.sqrt(torch.sum(graph_signal[:, :, 1:3] ** 2, dim=-1, keepdim=True))
        cosine_sims[cosine_sims <= 0] = 0

        adjacency_matrix = (cosine_sims * wind_speed) / self.distance
        adjacency_matrix = adjacency_matrix.view(num_samples, num_vertices, num_vertices)

        return adjacency_matrix


class cheb_conv_aagcn(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, fusiongraph, in_channels, out_channels, num_of_vertices, DEVICE):
        '''
        :param K: int
        :param in_channels: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_aagcn, self).__init__()
        self.K = K
        self.DEVICE = DEVICE  # 改动
        self.fusiongraph = fusiongraph

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)

        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        adj_for_run = self.fusiongraph()

        edge_idx, edge_attr = dense_to_sparse(adj_for_run)
        edge_idx_l, edge_attr_l = get_laplacian(edge_idx, edge_attr)

        L_tilde = to_dense_adj(edge_idx_l, edge_attr=edge_attr_l)[0]
        cheb_polynomials = cheb_polynomial_torch(L_tilde, self.K)

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = cheb_polynomials[k]  # (N,N)

                myspatial_attention = spatial_attention[:, k, :, :]
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                T_k_with_at = T_k.mul(myspatial_attention)  # (N,N)*(N,N) = (N,N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)

                # spatial_attention graph convolution,
                rhs = torch.matmul(T_k_with_at.permute(0, 2, 1), graph_signal)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return self.relu(torch.cat(outputs, dim=-1))


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channels: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class TemporalConvModule (nn.Module):

    def __init__(self, nb_time_filter, time_strides, num_of_timesteps, DEVICE):
        super(TemporalConvModule, self).__init__()
        self.gtu5 = GTU(nb_time_filter, time_strides, 5)
        self.gtu7 = GTU(nb_time_filter, time_strides, 7)
        self.gtu9 = GTU(nb_time_filter, time_strides, 9)
        self.gate = GatedFusion(num_of_timesteps)
        self.DEVICE = DEVICE

        self.fcmy = nn.Sequential(
            nn.Linear(3 * num_of_timesteps - 18, num_of_timesteps),
            nn.Dropout(0.05),
        )

    def forward(self, spatial_feature):

        x_conv = []
        x_conv.append(self.gtu5(spatial_feature))  # B,F,N,T-4
        x_conv.append(self.gtu7(spatial_feature))  # B,F,N,T-6
        x_conv.append(self.gtu9(spatial_feature))  # B,F,N,T-8

        time_conv = torch.cat(x_conv, dim=-1)
        time_conv = self.fcmy(time_conv)

        time_conv_final = self.gate(spatial_feature, time_conv)

        return time_conv_final


class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, batch_size):
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long).cuda()
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features,
                                                       self.nb_seq)
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long).cuda()
            pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class GatedFusion(nn.Module):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.FC_xs = nn.Linear(D, D)
        self.FC_xt = nn.Linear(D, D)
        self.FC_h = nn.Sequential(
            nn.ReLU(),
            nn.Linear(D, D)
        )

    def forward(self, HS, HG):
        XS = self.FC_xs(HS)
        XG = self.FC_xt(HG)

        gate = torch.sigmoid(XS + XG)

        H = gate * HS + (1 - gate) * HG
        H = self.FC_h(H)
        return H


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=3)
        # beta1 = beta * z
        return (beta * z).sum(3), beta


class JMGCN_block(nn.Module):
    def __init__(self, DEVICE, fusiongraph, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 num_of_vertices, num_of_timesteps, len_input, d_model, d_k, d_v, n_heads, position_vectors, distance):
        super(JMGCN_block, self).__init__()
        # ---fusion
        self.DEVICE = DEVICE
        self.to(DEVICE)
        self.fusiongraph = fusiongraph
        self.adj_for_run = self.fusiongraph()
        self.n_heads = n_heads
        # mask
        self.adj_stack = load_long_static_matrix(DEVICE)
        self.relu = nn.ReLU(inplace=True)
        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d))
        self.Embed = Embedding(num_of_vertices, d_model, num_of_d, 'S')

        self.stAttention = ST_Attention(DEVICE, d_model, d_k, d_v, K)
        self.cheb_conv_KAt = cheb_conv_kagcn(K, fusiongraph, in_channels, nb_chev_filter,
                                             num_of_vertices, position_vectors, distance, DEVICE)
        self.cheb_conv_AAt = cheb_conv_aagcn(K, fusiongraph, in_channels, nb_chev_filter, num_of_vertices, DEVICE)
        self.Gated_fusion = GatedFusion(num_of_timesteps)
        self.attention_fusion = Attention(nb_chev_filter)

        self.tcm_conv_module = TemporalConvModule(nb_time_filter, time_strides, num_of_timesteps, DEVICE)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                          return_indices=False, ceil_mode=False)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))

        self.dropout = nn.Dropout(p=0.05)
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, last_att):
        '''
        :param x: (Batch_size, N, F_in, T)
        :param res_att: (Batch_size, N, F_in, T)
        :return: (Batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape  # B,N,F,T

        x_att = self.pre_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        embed_spatial = self.Embed(x_att, batch_size)
        embed_spatial = self.dropout(embed_spatial)
        re_atten, st_atten = self.stAttention(embed_spatial, embed_spatial, self.adj_stack, last_att)  # B,Hs,N,N

        # graph convolution in spatial dim
        ka_gcn = self.cheb_conv_KAt(x, st_atten)   # B,N,F,T
        aa_gcn = self.cheb_conv_AAt(x, st_atten)  # Static
        # KAGCN and AAGCN, gated fusion
        spatial_gcn = self.Gated_fusion(ka_gcn, aa_gcn)

        # attention fusion
        # emb = torch.stack([ka_gcn.permute(0, 1, 3, 2), aa_gcn.permute(0, 1, 3, 2),], dim=3)
        # spatial_gcn, att = self.attention_fusion(emb)  # b, n, t, f
        # spatial_gcn = spatial_gcn.permute(0, 1, 3, 2)  # B, N, F, T

        # convolution along the time axis
        s_gcn = spatial_gcn.permute(0, 2, 1, 3)  # B,F,N,T
        tcm_conv = self.tcm_conv_module(s_gcn)
        # time_conv = self.time_conv(s_gcn)  # Ablation for TCM
        tcm_conv_output = self.relu(s_gcn + tcm_conv)  # B,F,N,T

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x_residual = self.ln(F.relu(x_residual + tcm_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual, re_atten


class JMGCN_submodule(nn.Module):
    def __init__(self, gpu_id, fusiongraph, in_channels, len_input, num_for_predict):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param num_for_predict:
        '''
        # Fusion Graph
        device = 'cuda:%d' % gpu_id
        DEVICE = device
        # -----------------
        # Parameters
        K = 2
        nb_block = 2
        nb_chev_filter = 32
        nb_time_filter = 32
        time_strides = 1

        num_of_timesteps = len_input
        num_of_vertices = fusiongraph.graph.node_num
        position_vectors, distance = compute_position_vectors(DEVICE)
        d_model = 512  # 512
        d_k = 32
        d_v = d_k
        num_of_d = in_channels
        n_heads = 3

        super(JMGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([JMGCN_block(DEVICE, fusiongraph, num_of_d, in_channels, K,
                                                      nb_chev_filter, nb_time_filter, time_strides,
                                                      num_of_vertices, num_of_timesteps, len_input, d_model, d_k, d_v, n_heads, position_vectors, distance)])

        self.BlockList.extend([JMGCN_block(DEVICE, fusiongraph, nb_time_filter, nb_chev_filter, K,
                                             nb_chev_filter, nb_time_filter, 1,
                                             num_of_vertices, num_of_timesteps, len_input // time_strides, d_model, d_k,
                                             d_v, n_heads, position_vectors, distance) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int((len_input / time_strides) * nb_block), 128, kernel_size=(1, nb_time_filter))
        self.final_fc = nn.Linear(128, num_for_predict)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        """
        for block in self.BlockList:
            x = block(x)
        """
        need_concat = []
        x = x.permute(0, 2, 3, 1)
        last_att = 0

        for block in self.BlockList:
            x, last_att = block(x, last_att)
            need_concat.append(x)

        final_x = torch.cat(need_concat, dim=-1)
        output1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        output = self.final_fc(output1).permute((0, 2, 1))[..., None]    # (b, t, n ,1)

        return output
