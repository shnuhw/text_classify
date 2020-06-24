#!/usr/bin/env python

from sklearn.metrics import classification_report, confusion_matrix


def evaluate(true_label_list, pre_label_list):

    report_result = classification_report(true_label_list, pre_label_list)
    confusion_result = confusion_matrix(true_label_list, pre_label_list)

    result_dict = {
        'report': report_result,
        'confusion': confusion_result
    }

    return result_dict
