# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 14:41
@Author: XINZHI YAO
"""


import torch
import torch.nn as nn
import torch.nn.functional as f


class BertClassifier(nn.Module):

    def __init__(self, bert, hidden_size, label_number, dropout):
        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(hidden_size, label_number)
        self.dropout = nn.Dropout(dropout)

        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.Softmax = nn.Softmax(dim=1)


    def forward(self, encoded_input, labels=None):

        bert_output = self.bert(**encoded_input)

        # batch_size, embedding_size
        cls_state = bert_output['last_hidden_state'][:, 0, :]

        # batch_size, embedding_size
        # dropout_layer = nn.Dropout(paras.dropout_prob)
        predicted = self.dropout(cls_state)

        # batch_size, label_number
        # fc_layer = nn.Linear(paras.hidden_size, paras.num_tags)
        predicted = self.fc(predicted)
        if labels is not None:
            loss = self.CrossEntropyLoss(predicted, labels)
            return loss
        else:
            predicted = self.Softmax(predicted)
            predicted_label = torch.argmax(predicted, dim=1)
            return predicted_label

