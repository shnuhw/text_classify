#!/usr/bin/env python
# author: hanwei
# email: hanwei@datagrand.com
# file: lstm.py
# time: 2020/6/11 18:59

import torch
from torch import nn

"vocab_size, input_size, out_dim, embedding_dim=300, hidden_size=256, num_layers=1"


class TextLSTM(nn.Module):

    def __init__(self, config):
        """

        :param config: Object
            vocab_size
            input_size
            out_dim
            embedding_dim=300
            hidden_size=256
            num_layers=1
        """
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=True)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(config.hidden_size*2*config.max_len, config.out_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedding = self.embedding(x)
        lstm_input = embedding.permute(1, 0, 2)
        # print(lstm_input.size())
        output, (h, c) = self.lstm(lstm_input)
        # print(output.size())
        linear_input = torch.cat([output[i, :, :] for i in range(output.shape[0])], dim=1)
        # print(linear_input.size())
        linear_output = self.linear(linear_input)
        final_output = self.softmax(linear_output)
        return final_output
