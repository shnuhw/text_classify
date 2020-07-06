#!/usr/bin/env python
# author: hanwei
# email: hanwei@datagrand.com
# file: data.py
# time: 2020/6/9 00:27

import torch
from torch.nn import functional as F
from torch import nn


class TextCNN(nn.Module):

    def __init__(self, config):
        """

        :param config: Object
            vocab_size
            out_dim
            embedding_dim=256,
            num_filters=128,
            filter_sizes=(2, 3, 4, 5),
            dropout=0.5
        """

        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.data.copy_(config.weight_matrix)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embedding_dim))
                                    for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.num_filters * len(config.filter_sizes), config.out_dim)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # print('input size:', x.size(), x.dtype)
        out = self.embedding(x)
        # print(out.size())
        out = out.unsqueeze(1)
        # print(out.size())
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
