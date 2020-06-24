#!/usr/bin/env python
# author: hanwei
# file: hyperparameter_config.py
# time: 2020/6/24 16:47


class TrainConfig:

    def __init__(self):
        self.optimizer = "adam"
        self.lr = 0.01
        self.epochs = 50
        self.batch_size = 64
        self.validation_split = 0.2
        self.early_stopping_patience = 8
        self.reduce_lr_factor = 0.1
        self.reduce_lr_patience = 2
        self.max_len = 256


class LSTMConfig:

    def __init__(self):
        self.embedding_dim = 300
        self.max_len = 20
        self.hidden_size = 256
        self.num_layers = 1


class TextCNNConfig:

    def __init__(self):
        self.embedding_dim = 300
        self.num_filter = 128
        self.filter_sizes = (2, 3, 4, 5)
        self.dropout = 0.5
        self.max_len = 20
