#!/usr/bin/env python
# author: hanwei
# email: hanwei@datagrand.com
# file: lstm.py
# time: 2020/6/11 18:59

import torch
from torch import nn
import torch.nn.functional as F


class TextLSTMAttention(nn.Module):

    def __init__(self, vocab_size, input_size, output_dim, embedding_dim=300, hidden_size=256, num_layers=1):
        super(TextLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(hidden_size*2, 64)
        self.linear2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2))

    def forward(self, x):
        embedding = self.embedding(x)
        # lstm_input = embedding.permute(1, 0, 2)
        output, (h, c) = self.lstm(embedding)
        # print(output.size())
        # lstm_out_reshape = output.permute(1, 0, 2)
        m = self.tanh(output)
        alpha = F.softmax(torch.matmul(m, self.w), dim=1).unsqueeze(-1)
        out = F.relu(torch.sum(output * alpha, 1))
        # linear_input = torch.cat([output[i, :, :] for i in range(output.shape[0])], dim=1)
        # print(linear_input.size())
        linear_output = self.linear(out)
        linear_output = self.linear2(linear_output)
        # final_output = self.softmax(linear_output)
        return linear_output
