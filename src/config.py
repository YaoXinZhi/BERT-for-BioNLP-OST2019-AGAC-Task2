# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 12/04/2021 11:06
@Author: XINZHI YAO
"""

import logging


class config:
    def __init__(self):

        # Original data
        self.data_path = '../data'
        self.train_json_path = '../data/AGAC_training/json'
        self.test_json_path = '../data/AGAC_answer/json'
        self.train_relation_statistics = '../data/train_relations.txt'
        self.test_relation_statistics = '../data/test_relations.txt'

        # Preprocessed data
        self.train_file = '../data/train.small.txt'
        self.test_file = '../data/test.small.txt'
        self.label_file = '../data/labels.txt'

        # Data loading parameters
        self.unknown_token = '[UNK]'
        self.max_sequence_length = 100

        # Model initialization parameters
        self.model_name = 'dmis-lab/biobert-base-cased-v1.1'
        self.hidden_size = 768
        self.label_number = 3
        self.dropout_prob = 0.3

        # Training parameters
        self.num_train_epochs = 20
        self.batch_size = 24
        self.shuffle = True
        self.drop_last = False

        # Optimizer parameters
        self.optimizer = 'Adam'
        self.learning_rate = 2e-5
        self.adam_epsilon = 1e-8

        # Logging parameters.
        self.save_log_file = True
        self.logging_level = logging.NOTSET
        self.log_save_path = '../logging'
        self.log_file = 'BertClassifier.log'
        self.log_file = 'log.4-13.txt'
        self.model_save_name = 'bert.4-13.pkl'

        # data preprocess
        self.special_tokens = ['<Var>', '']

