# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 15:43
@Author: XINZHI YAO
"""

import os
import time
import random
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer, BertModel
from transformers import AdamW

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from src.model import BertClassifier
# from utils import *
# from src.dataloader import RE_Dataset
# from src.config import config

from model import BertClassifier
from utils import *
from dataloader import RE_Dataset
from config import config


def evaluation(model, tokenizer, dataloader, max_sequence_length: int, label_to_index: dict):
    model.eval()
    total_pred_label = []
    total_ture_label = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):

            batch_data, batch_label = batch

            label_idx = batch_label_to_idx(batch_label, label_to_index, False)
            encoded_data = tokenizer(batch_data,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt',
                                     max_length=max_sequence_length)


            predict_label = model(encoded_data)
            predict_label = tensor_to_list(predict_label)

            total_ture_label.extend(label_idx)
            total_pred_label.extend(predict_label)

        acc = accuracy_score(total_ture_label, total_pred_label)
        precision = precision_score(total_ture_label, total_pred_label, average='macro')
        recall = recall_score(total_ture_label, total_pred_label, average='macro')
        f1 = f1_score(total_ture_label, total_pred_label, average='macro')

    return acc, precision, recall, f1


def main(paras):

    logger = logging.getLogger(__name__)
    if paras.save_log_file:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=paras.logging_level,
                            filename=f'{paras.log_save_path}/{paras.log_file}',
                            filemode='w')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=paras.logging_level, )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f'Loading model: {paras.model_name}')
    tokenizer = BertTokenizer.from_pretrained(paras.model_name)
    bert = BertModel.from_pretrained(paras.model_name)

    train_dataset = RE_Dataset(paras, 'train')
    train_dataloaer = DataLoader(train_dataset, batch_size=paras.batch_size,
                                 shuffle=paras.shuffle, drop_last=paras.drop_last)
    label_to_index = train_dataset.label_to_index

    test_dataset = RE_Dataset(paras, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=paras.batch_size,
                                 shuffle=paras.shuffle, drop_last=paras.drop_last)

    bert_classifier = BertClassifier(bert, paras.hidden_size, paras.label_number,
                                     paras.dropout_prob)

    if paras.optimizer == 'adam':
        logger.info('Loading Adam optimizer.')
        optimizer = torch.optim.Adam(bert_classifier.parameters(), lr=paras.learning_rate)
    elif paras.optimizer == 'adamw':
        logger.info('Loading AdamW optimizer.')
        no_decay = [ 'bias', 'LayerNorm.weight' ]
        optimizer_grouped_parameters = [
            {'params': [ p for n, p in bert_classifier.named_parameters() if not any(nd in n for nd in no_decay) ],
             'weight_decay': 0.01},
            {'params': [ p for n, p in bert_classifier.named_parameters() if any(nd in n for nd in no_decay) ],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=paras.learning_rate,
                          eps=args.adam_epsilon)
    else:
        logger.warning(f'optimizer must be "Adam" or "AdamW", but got {paras.optimizer}.')
        logger.info('Loading Adam optimizer.')
        optimizer = torch.optim.Adam(bert_classifier.parameters(),
                                     lr=paras.learning_rate)


    logger.info('Training Start.')
    best_eval = {'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss': 0}
    for epoch in range(paras.num_train_epochs):
        epoch_loss = 0
        bert_classifier.train()
        for step, batch in enumerate(train_dataloaer):
            optimizer.zero_grad()

            batch_data, batch_label = batch

            encoded_data = tokenizer(batch_data,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt',
                                     max_length=paras.max_sequence_length)

            label_tensor = batch_label_to_idx(batch_label, label_to_index)

            loss = bert_classifier(encoded_data, label_tensor)

            epoch_loss += loss_to_int(loss)

            logging.info(f'epoch: {epoch}, step: {step}, loss: {loss:.4f}')

            # fixme: del
            # acc, precision, recall, f1 = evaluation(bert_classifier, tokenizer, test_dataloader,
            #                                         paras.max_sequence_length, label_to_index)
            # logger.info(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, '
            #             f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_dataloaer)

        acc, precision, recall, f1 = evaluation(bert_classifier, tokenizer, test_dataloader,
                                                paras.max_sequence_length, label_to_index)

        logging.info(f'Epoch: {epoch}, Epoch-Average Loss: {epoch_loss:.4f}')
        logger.info(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, '
                    f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

        if best_eval['loss'] == 0 or f1 > best_eval['f1']:
            best_eval['loss'] = epoch_loss
            best_eval['acc'] = acc
            best_eval['precision'] = precision
            best_eval['recall'] = recall
            best_eval['f1'] = f1
            torch.save(bert_classifier, f'{paras.log_save_path}/{paras.model_save_name}')

            with open(f'{paras.log_save_path}/checkpoint.log', 'w') as wf:
                wf.write(f'Save time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
                wf.write(f'Best F1-score: {best_eval["f1"]:.4f}\n')
                wf.write(f'Precision: {best_eval["precision"]:.4f}\n')
                wf.write(f'Recall: {best_eval["recall"]:.4f}\n')
                wf.write(f'Accuracy: {best_eval["acc"]:.4f}\n')
                wf.write(f'Epoch-Average Loss: {best_eval["loss"]:.4f}\n')

            logger.info(f'Updated model, best F1-score: {best_eval["f1"]:.4f}\n')

    logger.info(f'Train complete, Best F1-score: {best_eval["f1"]:.4f}.')

if __name__ == '__main__':
    paras = config()

    main(paras)
