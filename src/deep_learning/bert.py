#!/usr/bin/env python
# author: hanwei
# file: bert.py
# time: 2020/7/6 17:04

# from transformers import BertTokenizer
import os

import torch
from torch import nn

# from transformers import AutoTokenizer, AutoModelWithLMHead
# from transformers import BertModel
# tokenizer = AutoTokenizer.from_pretrained("./bert_model/bert-base-chinese/")


class Bert(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.bert = BertModel.from_pretrained('./bert_model/bert-base-chinese/')
        self.bert = config.bert
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, 20)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        return out
