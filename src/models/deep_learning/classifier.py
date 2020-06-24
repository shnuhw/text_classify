#!/usr/bin/env python
# author: hanwei
# file: train.py
# time: 2020/6/23 11:38

import os
import pickle

from torch import nn, optim
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from dataset import DataSet


class Classifier:

    def __init__(self, net, config, data_set, model_dir, is_train=True):
        self.net = net
        self.config = config
        self.model_dir = model_dir
        self.dataset = data_set
        self.model_path = self.model_dir + '/model.model'
        self.vocab_path = self.model_dir + '/vocab.pkl'
        if is_train:
            self.model = None
            self.vocab = None
        else:
            self.model = self.net.load_state_dict(torch.load(self.model_path)) 
            self.vocab = pickle.load(open(self.vocab_path, 'rb'))

    def train(self):

        self.net.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        train_iter, test_iter, eval_iter = self.dataset.get_batch_data()
        pickle.dump(dataset.vocab, open(self.vocab_path, 'wb'))

        print('training...')
        batch_count = 0
        num_epochs = self.config.num_epoch
        dev_best_loss = float('inf')
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            for index, train_item in enumerate(train_iter):
                inputs = train_item.title
                labels = train_item.label

                # zero the parameter gradients
                optimizer.zero_grad()
                # print(inputs.dtype)
                if self.config.cuda:
                    outputs = self.net(inputs.cuda())
                else:
                    outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_count % 100 == 0:
                    true = labels.data
                    predict = torch.max(outputs.data, 1)[1]
                    train_acc = accuracy_score(true.cpu(), predict.cpu())
                    dev_acc, dev_loss = self.evaluate(test_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(
                            self.net.state_dict(), self.model_path)

                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  ' \
                          'Val Acc: {4:>6.2%}'
                    print(
                        msg.format(
                            batch_count,
                            loss.item(),
                            train_acc,
                            dev_loss,
                            dev_acc))

                    self.net.train()
                batch_count += 1

    def evaluate(self, data_iter):
        self.net.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for item in data_iter:
                texts = item.title
                labels = item.label
                if self.config.cuda:
                    outputs = self.net(texts.cuda())
                else:    
                    outputs = self.net(texts)
                loss = nn.functional.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = accuracy_score(labels_all, predict_all)
        # if test:
        #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        #     confusion = metrics.confusion_matrix(labels_all, predict_all)
        #     return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)

    def predict(self, text, cuda=False):
        if cuda:
            text_vec = torch.tensor([[self.vocab.stoi[char] for char in text]], dtype=torch.long).cuda()
        else:
            text_vec = torch.tensor([[self.vocab.stoi[char] for char in text]], dtype=torch.long)
        with torch.no_grad():
            outputs = self.net(text_vec)
            # labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            return predic[0]


class Config:

    def __init__(self):
        self.max_len = 20
        self.batch_size = 64
        self.root_dir = '../../../data'
        self.train_file_name = 'train.csv'
        self.test_file_name = 'test.csv'
        self.eval_file_name = 'valid.csv'
        self.num_epoch = 20
        self.cuda = False


if __name__ == '__main__':
    from text_cnn import TextCNN
    from lstm import TextLSTM
    config = Config()
    dataset = DataSet(config.max_len, config.batch_size, config.root_dir,
                      config.train_file_name,
                      config.test_file_name,
                      config.eval_file_name,
                      )
    num_label = len(dataset.label_vocab)
    vocab_size = len(dataset.vocab)
    # cnn = TextCNN(vocab_size, num_label).cuda()
    lstm = TextLSTM(vocab_size, config.max_len, num_label)
    clf = Classifier(
        lstm,
        config,
        dataset,
        model_dir='../../../model/test_lstm/',
        is_train=True)
    clf.train()

    # text = '中石化大涨7% 唱的是哪出戏'
    # import time
    # time_start = time.time()
    # r = clf.predict(text, True)
    # time_cost = time.time() - time_start
    # print(time_cost)
    # print(dataset.label_vocab.itos[r])
