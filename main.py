#!/usr/bin/env python

from src.models.machine_learning.classifier import Classifier


def main():
    train_file_path = './data/train.csv'
    eval_file_path = './data/test.csv'
    model_dir = './model/test/'
    clf = Classifier(train_file_path, eval_file_path, model_dir, is_train=True)
    clf.train(clf_model='lr')
    print(clf.predict(['希腊援助贷款存变数 黄金温和收高'], method='predict_proba'))
    clf.eval()


if __name__ == '__main__':
    main()
