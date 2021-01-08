#!/usr/bin/env python
# author: hanwei
# file: dataset.py
# time: 2020/6/11 23:57

import pandas as pd
from torchtext import data
from torchtext.vocab import GloVe, Vectors
import torch
import pickle


class DataSet:

    def __init__(self, max_length, batch_size, w2v_file_path, w2v_cache_path, root_dir_path,
                 train_file_name='train.csv', test_file_name='test.csv',
                 val_file_name='valid.csv', tokenizer=None):
        self.max_len = max_length
        self.batchsize = batch_size
        self.train_dataset = None
        self.test_dataset = None
        self.eval_dataset = None
        self.vocab = []
        self.embedding = {}
        self._init_dataset(
            root_dir_path,
            train_file_name,
            test_file_name,
            val_file_name,
            w2v_file_path,
            w2v_cache_path
        )
        self.tokenizer = tokenizer

    def _init_dataset(self, root_dir_path, train_file_path, test_file_path,
                      val_file_path, w2v_file_path=None, w2v_cache_path=None):
        def tokenizer(text):
            return [word for word in text]

        if self.tokenizer:
            PAD_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            UNK_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
            TEXT = data.Field(sequential=True, tokenize=self.tokenizer.encode, lower=True, use_vocab=False,
                              batch_first=True, fix_length=self.max_len, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        else:
            TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True,
                              batch_first=True,
                              fix_length=self.max_len)
        LABLE = data.Field(sequential=False, use_vocab=True)
        datafields = [(None, None), ("title", None), ("text", TEXT),
                      ("label", LABLE), ("file_name", None)]
        self.train_dataset, self.val_dataset, self.test_dataset = data.TabularDataset.splits(path=root_dir_path,
                                                                                             train=train_file_path,
                                                                                             validation=val_file_path,
                                                                                             test=test_file_path,
                                                                                             format='csv',
                                                                                             fields=datafields)
        vectors = Vectors(
            name=w2v_file_path,
            cache=w2v_cache_path
        )
        TEXT.build_vocab(self.train_dataset, vectors=vectors)
        LABLE.build_vocab(self.train_dataset)
        self.vocab = TEXT.vocab
        self.label_vocab = LABLE.vocab
        # pickle.dump(self.vocab, open('./test.pkl', 'wb'))

    def get_batch_data(self):
        train_iter, eval_iter, test_iter = data.BucketIterator.splits((self.train_dataset,
                                                                       self.val_dataset,
                                                                       self.test_dataset),
                                                                      batch_sizes=(self.batchsize,
                                                                                   self.batchsize,
                                                                                   self.batchsize),
                                                                      sort_key=lambda x: len(
                                                                          x.text),
                                                                      sort_within_batch=True,
                                                                      repeat=False,
                                                                      shuffle=True
                                                                      )
        return train_iter, eval_iter, test_iter

    def get_single_text(self):
        pass

    def get_batch_text(self):
        pass


if __name__ == '__main__':
    ds = DataSet(
        20,
        10,
        '../../../data/',
        'train.csv',
        'test.csv',
        'valid.csv')

    a, b, c = ds.get_batch_data()
    # pickle.dump(ds, open('./dataset.pkl', 'wb'))
    # d = pickle.load(open('./dataset.pkl', 'rb'))
    # print(d.vocab.stoi['中'])
    for i, item in enumerate(a):
        print(item.title)
        print(item.label)
        break

    # vocab = pickle.load(open('./test.pkl', 'rb'))
    # print(len(vocab))
    # print(vocab.stoi['中'])
