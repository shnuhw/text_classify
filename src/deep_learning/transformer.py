#!/usr/bin/env python
# author: hanwei
# file: transformer.py
# time: 2020/6/28 17:58

import torch
from torch import nn
import numpy as np
import math


class Transformer(nn.Module):

    def __init__(self, config):
        """

        :param config: Object
        vocab_size, max_len, embedding_dim, num_head, hidden_size, encoder_num, out_dim
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(config.vocab_size, config.max_len,
                               config.embedding_dim, config.num_head, config.hidden_size, config.device)
        self.fc1 = nn.Linear(config.embedding_dim * config.embedding_dim, config.embedding_dim)
        self.fc2 = nn.Linear(config.embedding_dim, config.out_dim)
        self.encoder_num = config.encoder_num
        self.embedding_dim = config.embedding_dim
        self.device = config.device

    def forward(self, x):

        out = self.encoder(x, True)

        for i in range(self.encoder_num-1):
            out = self.encoder(out, False)

        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class Encoder(nn.Module):

    def __init__(self, vocab_size, seq_len, embedding_dim, num_head, hidden_size, device):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = PositionEncoding(seq_len, embedding_dim, device)
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

    def __init__(self, seq_len, embedding_dim, device):
        super(PositionEncoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000 ** (2 * i / embedding_dim))
                                 for i in range(embedding_dim)]
                                for pos in range(seq_len)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.device = device

    def forward(self, x):

        # assert x.size() == self.pe.size()
        # print(x.size(), self.pe.size())

        return x + nn.Parameter(self.pe, requires_grad=False).to(self.device)


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
        self.embedding_dim = embedding_dim

        self.attention = ScaledDotProductAttention(self.head_dim)

        self.fc_last = nn.Linear(embedding_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        Q = self.fc_WQ(x)
        K = self.fc_WK(x)
        V = self.fc_WV(x)
        attention_list = []
        for q, k, v in zip(self.head_WQ, self.head_WK, self.head_WV):
            attention_list.append(self.attention(q(Q), k(K), v(V)))
        Z = torch.cat(attention_list, dim=2)
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
