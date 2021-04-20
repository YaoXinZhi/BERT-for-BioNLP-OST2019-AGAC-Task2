# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 15:45
@Author: XINZHI YAO
"""

import torch

def batch_label_to_idx(batch_label: list, label_to_index: dict,
                       return_tensor=True):

    label_idx = [label_to_index[label] for label in batch_label]

    if return_tensor:
        return torch.LongTensor(label_idx)
    else:
        return label_idx

def loss_to_int(loss):
    return loss.detach().cpu().item()

def tensor_to_list(tensor):
    return tensor.numpy().tolist()

def input_ids_to_token(tokenizer, input_ids):

    token_list = [''.join(tokenizer.decode(idx).split()) for idx in input_ids]
    return token_list



