# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 9:57
@Author: XINZHI YAO
"""

import logging

from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

# todo: convert logger to
logger = logging.getLogger(__name__)


class RE_Dataset(Dataset):
    def __init__(self, paras, mode: str):
        if mode == 'train':
            print('loading train dataset.')
            self.data_path = paras.train_file
        elif mode == 'test':
            print('loading test dataset.')
            self.data_path = paras.test_file
        else:
            raise ValueError(f'mode must be "train" or "test",'
                             f' but got "{mode}".')

        self.label_file = paras.label_file

        self.data_statistics = defaultdict(int)
        self.data = []
        self.label = []

        self.label_set = set()
        self.label_to_index = {}
        self.index_to_label = {}

        self.read_label()
        self.read_data()

        if len(self.data_statistics.keys()) > 3:
            print('Too many relations in data set.')
            print(self.data_statistics.keys())

        # print(f'{self.data_statistics}')
        themeof_count = self.data_statistics['ThemeOf']
        causeof_count = self.data_statistics['CauseOf']
        norelation_count = self.data_statistics['NoRelation']

        print(f'Positive: {themeof_count+causeof_count} (ThemeOf: {themeof_count}, '
              f'CauseOf: {causeof_count}), '
              f'Negative: {norelation_count}.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def read_label(self):

        with open(self.label_file) as f:
            for line in f:
                label = line.strip()
                self.label_set.add(label)

        self.label_to_index = {label: idx for idx, label
                               in enumerate(self.label_set)}
        self.index_to_label = {idx: label for idx, label
                               in enumerate(self.label_set)}

    def read_data(self):
        with open(self.data_path, encoding='utf-8') as f:
            f.readline()
            for line in f:
                l = line.strip().split('\t')
                token1, label1, _ = l[0], l[1], l[2]
                token2, label2, _ = l[3], l[4], l[5]
                relation, sentence = l[6], l[7]

                data = f'[CLS]{sentence}[SEP]' \
                       f'{token1}[SEP]{label1}[SEP]' \
                       f'{token2}[SEP]{label2}[SEP]'

                self.data_statistics[relation] += 1
                self.data.append(data)
                self.label.append(relation)


    def print_example(self):
        print('Positive Example:')
        for i in range(len(self.data)):
            if self.label[i] == 'CauseOf' or self.label[i] == 'ThemeOf':
                print(f'{self.data[i]}\t{self.label[i]}')
                break
        print()
        print('Negative Example:')
        for i in range(len(self.data)):
            if self.label[i] == 'NoRelation':
                print(f'{self.data[i]}\t{self.label[i]}')
                break

# todo: Anonymously


if __name__ == '__main__':
    pass

    # args = config()
    #
    # train_dataset = RE_Dataset(args, 'train', )
    #
    # print('Positive Example:')
    # print(train_dataset[0])
    #
    # print()
    # print('Negative Example:')
    # for i in train_dataset:
    #     if i[-1] == 'NoRelation':
    #         print(i)
    #         break


