#!/usr/bin/env python
# author: hanwei
# file: lr.py
# time: 2020/6/1 18:32

import joblib
import time

import pandas as pd
import jieba
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from ..evaluate import evaluate

feature_select_dict = {
    'count': CountVectorizer,
    'tf_idf': TfidfVectorizer
}

model_dict = {
    'svm': SVC,
    'lr': LogisticRegression,
    'xgb': xgb,
    'rf': RandomForestClassifier
}


class Classifier:

    def __init__(self, train_file_path, eval_file_path, model_dir, is_train=True):
        self.train_file_path = train_file_path
        self.eval_file_path = eval_file_path
        self.model_dir = model_dir
        self.model_path = model_dir + '/model.model'
        if is_train:
            self.clf = None
        else:
            self.load_mode()

    def load_mode(self):
        self.clf = joblib.load(self.model_path)

    @staticmethod
    def get_one_feature(text, split='word'):
        assert split in ['word', 'char'], 'ERROR: \'split\' not in word and char'
        if split == 'word':
            return ' '.join([word for word in jieba.cut(text)])
        elif split == 'char':
            return ' '.join([char for char in text])

    def get_features(self):
        feature_list = []
        label_list = []
        df = pd.read_csv(self.train_file_path)
        for index, row in df.iterrows():
            title = row['title']
            label = row['label']
            if not isinstance(title, str):
                continue
            feature_list.append(self.get_one_feature(title, split='char'))
            label_list.append(label)

        return feature_list, label_list

    def train(self, vector='tf_idf', clf_model='lr'):
        assert vector in feature_select_dict
        assert clf_model in model_dict
        train_info_file = self.model_dir + '/train.info'
        f_train_info = open(train_info_file, 'w')
        feature_vectorized = feature_select_dict[vector]
        if clf_model == 'xgb':
            classify_model = model_dict[clf_model].XGBClassifier
        else:
            classify_model = model_dict[clf_model]

        X, y = self.get_features()
        f_train_info.write('Train file length: {} \n'.format(len(X)))
        vec = feature_vectorized(analyzer='char', ngram_range=(1, 2))
        classifier = classify_model()
        self.clf = Pipeline([
            ('vectorized', vec),
            ('clf', classifier)
         ])
        time_start = time.time()
        self.clf.fit(X, y)
        f_train_info.write('Feature length: {}\n'.format(len(self.clf.named_steps.get('vectorized').vocabulary_)))
        # sample_features = []
        feature_dict = self.clf.named_steps.get('vectorized').vocabulary_
        # print(type(feature_dict.keys()))

        sample_features = []
        for index, feature in enumerate(feature_dict.keys()):
            if index > 50:
                break
            sample_features.append(feature)
        f_train_info.write('Sample features: {}\n'.format('\n'.join(sample_features)))

        time_end = time.time()
        f_train_info.write('Train time cost: {}s\n'.format(time_end - time_start))
        joblib.dump(self.clf, self.model_path)
        f_train_info.close()

    def predict(self, text_list, method='predict'):

        predict_list = []
        if method == 'predict':
            pre_list = self.clf.predict([self.get_one_feature(text, split='char') for text in text_list])
            return pre_list
        elif method == 'predict_proba':
            class_list = self.clf.named_steps['clf'].classes_
            pre_list = self.clf.predict_proba([self.get_one_feature(text, split='char') for text in text_list])
            for result_list in pre_list:
                item_result_dict = {}
                for class_name, score in zip(class_list, result_list):
                    item_result_dict.setdefault(class_name, score)
                predict_list.append(item_result_dict)
            return predict_list

    def eval(self):
        # todo：评估模型，指标包括：各个类别的P、R、F1，宏平均、微平均、加权宏平均以及混淆矩阵
        eval_file = self.model_dir + '/eval.info'
        f_eval_info = open(eval_file, 'w')
        df = pd.read_csv(self.eval_file_path)
        df = df.dropna(how='any')
        text_list = df['title'].tolist()
        true_list = df['label'].tolist()
        pre_list = self.predict(text_list, method='predict')
        eval_result = evaluate(true_list, pre_list)
        f_eval_info.write(eval_result['report'])
        f_eval_info.write(str(eval_result['confusion']))
        f_eval_info.close()



