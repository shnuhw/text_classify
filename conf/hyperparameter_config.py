#!/usr/bin/env python
# author: hanwei
# file: hyperparameter_config.py
# time: 2020/6/24 16:47


class TrainConfig:

    def __init__(self):
        self.optimizer = "adam"
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 128
        self.validation_split = 0.2
        self.early_stopping_patience = 8
        self.reduce_lr_factor = 0.1
        self.reduce_lr_patience = 2
        self.max_len = 512
        self.train_file_dir = './data'
        self.model_dir = './model/test_lstm_attention'
        self.cuda = True


class LSTMConfig:

    def __init__(self):
        self.embedding_dim = 300
        self.max_len = 512
        self.hidden_size = 256
        self.num_layers = 1


class LSTMAttentionConfig:

    def __init__(self):
        self.embedding_dim = 300
        self.max_len = 512
        self.hidden_size = 256
        self.num_layers = 1
        self.dropout = 0.2


class TextCNNConfig:

    def __init__(self):
        self.embedding_dim = 300
        self.num_filters = 128
        self.filter_sizes = (2, 3, 4, 5)
        self.dropout = 0.5
        self.max_len = 512


class TransformerConfig:

    def __init__(self):
        self.embedding_dim = 512
        self.num_head = 8
        self.hidden_size = 256
        self.encoder_num = 2
        self.max_len = 512
