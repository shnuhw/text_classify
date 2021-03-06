#!/usr/bin/env python

import torch

from src.deep_learning.classifier import Classifier as DlClassifier
from src.deep_learning import dataset
from conf.net_select import net_dict

device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")


def main(net_name):
    print(net_name)
    assert net_name in net_dict

    train_conf = net_dict['train_conf']()
    train_conf.device = device

    net_class = net_dict[net_name]['net']
    net_config = net_dict[net_name]['conf']()

    mydataset = dataset.DataSet(train_conf.max_len,
                                train_conf.batch_size,
                                train_conf.w2v_file_path,
                                train_conf.w2v_cache_path,
                                train_conf.train_file_dir,
                                tokenizer=net_config.tokenizer)
    num_label = len(mydataset.label_vocab)
    vocab_size = len(mydataset.vocab)

    net_config.vocab_size = vocab_size
    net_config.out_dim = num_label
    net_config.device = device
    net_config.weight_matrix = mydataset.vocab.vectors

    net = net_class(net_config).to(device)
    clf = DlClassifier(
        net,
        mydataset,
        train_conf,
        is_train=True)
    for name, parameters in clf.net.named_parameters():
        print(name, ':', parameters.size())
    clf.train()


if __name__ == '__main__':
    import sys
    net_name = sys.argv[1]
    main(net_name)
