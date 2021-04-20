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

logger = logging.getLogger(__name__)


class RE_Dataset(Dataset):
    def __init__(self, paras, mode: str):
        if mode == 'train':
            logger.info('loading train dataset.')
            self.data_path = paras.train_file
        elif mode == 'test':
            logger.info('loading test dataset.')
            self.data_path = paras.test_file
        else:
            raise ValueError(f'mode must be "train" or "test",'
                             f' but got "{mode}".')

        self.ner_label_file = paras.ner_label_file
        self.special_token_set = set()

        self.load_denotation = paras.load_denotation
        self.add_denotation_span = paras.add_denotation_span

        self.label_file = paras.label_file

        self.data_statistics = defaultdict(int)
        self.data = []
        self.label = []

        self.label_set = set()
        self.label_to_index = {}
        self.index_to_label = {}

        self.load_special_token()
        self.read_label()
        self.read_data()

        if len(self.data_statistics.keys()) > 3:
            logger.warning('Too many relations in data set.')
            logger.warning(self.data_statistics.keys())

        themeof_count = self.data_statistics['ThemeOf']
        causeof_count = self.data_statistics['CauseOf']
        norelation_count = self.data_statistics['NoRelation']

        logger.info(f'Positive: {themeof_count+causeof_count} (ThemeOf: {themeof_count}, '
              f'CauseOf: {causeof_count}), '
              f'Negative: {norelation_count}.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def load_special_token(self):
        with open(self.ner_label_file) as f:
            for line in f:
                ner_label = line.strip()
                self.special_token_set.add(f'<S:{ner_label}>')
                self.special_token_set.add(f'</S:{ner_label}>')

    @staticmethod
    def add_span_token(sentence, label1, offset1, label2, offset2):

        offset_to_label = {offset1: label1, offset2: label2}

        first_offset, second_offset = sorted([offset1, offset2])

        first_label = offset_to_label[first_offset]
        second_label = offset_to_label[second_offset]

        first_head, first_tail = f'<S:{first_label}>', f'</S:{first_label}>'
        second_head, second_tail = f'<S:{second_label}>', f'</S:{second_label}>'

        sent_list = [s for s in sentence]
        sent_list.insert(second_offset[1], second_tail)
        sent_list.insert(second_offset[0], second_head)

        sent_list.insert(first_offset[1], first_tail)
        sent_list.insert(first_offset[0], first_head)

        sent = ''.join(sent_list)
        return sent


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

                token1, label1, offset1 = l[0], l[1], l[2]
                token2, label2, offset2 = l[3], l[4], l[5]
                relation, sentence = l[6], l[7]

                offset1 = eval(offset1)
                offset2 = eval(offset2)
                # print(token1,label1, token2,label2, sentence, offset1, offset2)

                # fixme: check the correctness of the token offset
                sentence_token = sentence[int(offset1[0]): int(offset1[1])].strip()
                if token1 != sentence_token:
                    print(sentence)
                    print(offset1)
                    print(f'{token1}-{len(token1)}-{sentence_token}-{len(sentence_token)}')
                    print()

                # 4-20 delete add [CLS] and [SEP] in the head and tail
                if self.load_denotation:
                    if self.add_denotation_span:
                        sentence = self.add_span_token(sentence, label1, offset1,
                                                       label2, offset2)
                        # data = f'[CLS]{sentence}[SEP]'
                        data = f'{sentence}'
                    else:
                        # data = f'[CLS]{sentence}[SEP]' \
                        #        f'{token1}[SEP]{label1}[SEP]' \
                        #        f'{token2}[SEP]{label2}[SEP]'
                        data = f'{sentence}[SEP]' \
                               f'{token1}[SEP]{label1}[SEP]' \
                               f'{token2}[SEP]{label2}'

                else:
                    # data = f'[CLS]{sentence}[SEP]' \
                    #        f'{token1}[SEP]' \
                    #        f'{token2}[SEP]'
                    data = f'{sentence}[SEP]' \
                           f'{token1}[SEP]' \
                           f'{token2}'

                self.data_statistics[relation] += 1
                self.data.append(data)
                self.label.append(relation)


    def print_example(self):
        logger.info('Positive Example:')
        for i in range(len(self.data)):
            if self.label[i] == 'CauseOf' or self.label[i] == 'ThemeOf':
                logger.info(f'{self.data[i]}\t{self.label[i]}')
                break
        logger.info('Negative Example:')
        for i in range(len(self.data)):
            if self.label[i] == 'NoRelation':
                logger.info(f'{self.data[i]}\t{self.label[i]}')
                break

    # todo: Anonymously


if __name__ == '__main__':
    # pass

    # from src.config import config
    args = config()
    #
    train_dataset = RE_Dataset(args, 'train', )
    #
    print('Positive Example:')
    print(train_dataset[26])
    #
    # print()
    print('Negative Example:')
    for i in train_dataset:
        if i[-1] == 'NoRelation':
            print(i)
            break


