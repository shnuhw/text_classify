#!/usr/bin/env python
# author: hanwei
# file: transformer.py
# time: 2020/6/28 17:58

import torch
from torch import nn
import numpy as np
import math
import copy


class Transformer(nn.Module):

    def __init__(self, vocab_size, seq_len, embedding_dim, num_head, hidden_size, encoder_num, out_dim, batch_size=64):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, seq_len, embedding_dim, num_head, hidden_size)
        self.fc = nn.Linear(embedding_dim * embedding_dim, out_dim)
        self.encoder_num = encoder_num
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

    def forward(self, x):

        out = self.encoder(x, True)

        for i in range(self.encoder_num-1):
            out = self.encoder(out, False)
        
        # print(out.size(), '11111111111')
        out = out.view(out.size()[0], -1) 
        # print(out.size(), '11111111111')
        out = self.fc(out)
        # print(out.size(), '22222222222')

        return out


class Encoder(nn.Module):

    def __init__(self, vocab_size, seq_len, embedding_dim, num_head, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = PositionEncoding(seq_len, embedding_dim)
        self.multi_head_atten = MultiHeadAttention(embedding_dim, num_head)
        self.position_wise_feed_forward = PositionWiseFeedForward(embedding_dim, hidden_size)

    def forward(self, x, is_zero=False):
        if is_zero:
            embedding = self.embedding(x)
            out = self.position_encoding(embedding)
        else:
            out = x
        out = self.multi_head_atten(out)
        out = self.position_wise_feed_forward(out)

        return out


class PositionEncoding(nn.Module):

    def __init__(self, seq_len, embedding_dim):
        super(PositionEncoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000 ** (2 * i / embedding_dim))
                                for i in range(embedding_dim)]
                               for pos in range(seq_len)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

    def forward(self, x):

        # assert x.size() == self.pe.size()
        # print(x.size(), self.pe.size())

        return x + nn.Parameter(self.pe, requires_grad=False).to('cuda')


class ScaledDotProductAttention(nn.Module):

    def __init__(self, k_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

        self.k_dim = k_dim

    def forward(self, K, Q, V):
        out = self.softmax(torch.matmul(Q, K.permute(0, 2, 1))/math.sqrt(self.k_dim))
        out = torch.matmul(out, V)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_head == 0
        self.fc_WQ = nn.Linear(embedding_dim, embedding_dim)
        self.fc_WK = nn.Linear(embedding_dim, embedding_dim)
        self.fc_WV = nn.Linear(embedding_dim, embedding_dim)

        self.num_head = num_head
        self.head_dim = embedding_dim // self.num_head
        self.head_WQ = nn.ModuleList([nn.Linear(embedding_dim, self.head_dim) for i in range(self.num_head)])
        self.head_WK = nn.ModuleList([nn.Linear(embedding_dim, self.head_dim) for i in range(self.num_head)])
        self.head_WV = nn.ModuleList([nn.Linear(embedding_dim, self.head_dim) for i in range(self.num_head)])
        # self.head_WK = nn.Linear(embedding_dim, self.head_dim)
        # self.head_WV = nn.Linear(embedding_dim, self.head_dim)
        self.embedding_dim = embedding_dim

        self.attention = ScaledDotProductAttention(self.head_dim)

        self.fc_last = nn.Linear(embedding_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        Q = self.fc_WQ(x)
        K = self.fc_WK(x)
        V = self.fc_WV(x)
        # q_head_list = [nn.Linear(self.embedding_dim, self.head_dim).cuda()(Q) for i in range(self.num_head)]
        # k_head_list = [nn.Linear(self.embedding_dim, self.head_dim).cuda()(K) for i in range(self.num_head)]
        # v_head_list = [nn.Linear(self.embedding_dim, self.head_dim).cuda()(V) for i in range(self.num_head)]
        # q_head_list = [copy.deepcopy(self.head_WQ(Q)) for i in range(self.num_head)]
        # k_head_list = [copy.deepcopy(self.head_WK(K)) for i in range(self.num_head)]
        # v_head_list = [copy.deepcopy(self.head_WV(V)) for i in range(self.num_head)]
        attention_list = []
        for q, k, v in zip(self.head_WQ, self.head_WK, self.head_WV):
            attention_list.append(self.attention(q(Q), k(K), v(V)))
        # print(len(attention_list), attention_list[0].size(), '0000000')
        Z = torch.cat(attention_list, dim=2)
        # print(len(attention_list), Z.size(), '0000000')
        out = self.fc_last(Z)

        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden)
        self.fc2 = nn.Linear(hidden, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
