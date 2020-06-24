#!/usr/bin/env python
# author: hanwei
# email: hanwei@datagrand.com
# file: lstm.py
# time: 2020/6/11 18:59

import torch
from torch import nn


class TextLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, embedding_dim=128, n_hidden=1):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=n_hidden)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(256, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedding = self.embedding(x)
        lstm_input = embedding.permute(1, 0, 2)
        output, (h, c) = self.lstm(lstm_input)
        linear_input = torch.cat([output[i, :, :] for i in range(output.shape[0])], dim=1)
        linear_output = self.linear(linear_input)
        final_output = self.softmax(linear_output)
        return final_output
