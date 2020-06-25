#!/usr/bin/env python

from src.deep_learning.classifier import Classifier as DlClassifier
from src.machine_learning.classifier import Classifier as MlClassifier
from src.deep_learning import lstm, text_cnn, dataset
from conf.hyperparameter_config import TrainConfig, LSTMConfig, TextCNNConfig


def main():
    net_config = LSTMConfig()
    train_config = TrainConfig()
    mydataset = dataset.DataSet(train_config.max_len, train_config.batch_size, train_config.train_file_dir)
    num_label = len(mydataset.label_vocab)
    vocab_size = len(mydataset.vocab)
    # cnn = TextCNN(vocab_size, num_label).cuda()
    lstm_net = lstm.TextLSTM(vocab_size, net_config.max_len, num_label)
    clf = DlClassifier(
        lstm_net,
        train_config,
        mydataset,
        model_dir=train_config.model_file_dir,
        is_train=True)
    for name, parameters in clf.net.named_parameters():
        print(name, ':', parameters.size())
    clf.train()


if __name__ == '__main__':
    main()
