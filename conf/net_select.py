#!/usr/bin/env python
# author: hanwei
# file: net_select.py
# time: 2020/7/1 22:12

from src.deep_learning import lstm_attention, lstm, text_cnn, transformer, bert
from .hyperparameter_config import *    # import all conf

net_dict = {
    'textCnn': {
        'net': text_cnn.TextCNN,
        'conf': TextCNNConfig
    },
    'lstm': {
        'net': lstm.TextLSTM,
        'conf': LSTMConfig
    },
    'lstmAttention': {
        'net': lstm_attention.TextLSTMAttention,
        'conf': LSTMAttentionConfig
    },
    'transformer': {
        'net': transformer.Transformer,
        'conf': TransformerConfig
    },
    'bert':{
        'net': bert.Bert,
        'conf': BertConfig
    },
    'train_conf': TrainConfig
}
