# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 15/04/2021 22:44
@Author: XINZHI YAO
"""


import os
import json
import spacy
import logging
from nltk import tokenize
from collections import defaultdict
from itertools import product, combinations


logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)

nlp = spacy.load('en_core_web_sm')


def get_token_offset(sent: str, token_list: list, ):
    token_end = 0
    token_to_offset = []
    for idx, token in enumerate(token_list):
        token_start = sent.index(token, token_end)
        token_end = token_start + len(token)
        token_to_offset.append((token,token_start, token_end, idx))
    return token_to_offset


def get_sent_offset(text: str):

    sent_list = tokenize.sent_tokenize(text)
    # doc = nlp(text)
    # sent_list = [sent.text for sent in doc.sents]

    begin = 0
    sent_to_offset = {}
    for sent in sent_list:
        end = begin + len(sent)
        sent_to_offset[sent] = (begin, end)
        begin = end + 1
    return sent_to_offset, sent_list


def denotation_sent_map(id_to_denotation: dict, sent_to_offset: dict):

    denotation_to_sent = {}
    sent_to_denotation = defaultdict(set)
    for d_id, (token, label, (t_begin, t_end)) in id_to_denotation.items():
        for sent, (s_begin, s_end) in sent_to_offset.items():
            if t_begin >= s_begin:
                if t_end <= s_end:
                    offset = (t_begin-s_begin, (t_begin-s_begin)+len(token))
                    denotation_to_sent[d_id] = (sent, offset)
                    sent_to_denotation[sent].add(d_id)
                    break
                else:
                    pass

    return denotation_to_sent, sent_to_denotation


def denotation_token_map(token_offset: list, denotation_offset: dict,
                         id_to_denotation: dict):
    denotation_to_idx = defaultdict(list)
    for d_id, (d_s, d_e) in denotation_offset.items():
        for (token, t_s, t_e, idx) in token_offset:
            if t_s <= d_s < t_e or t_s < d_e <= t_e:
                denotation_to_idx[d_id].append(idx)
        if len(denotation_to_idx[d_id]) == 1:
            denotation_to_idx[d_id].append(denotation_to_idx[d_id][0])
        elif len(denotation_to_idx[d_id]) > 2:
            denotation = denotation_to_idx[d_id]
            denotation_to_idx[d_id] = [denotation[0], denotation[-1]]
        denotation_to_idx[d_id].append(id_to_denotation[d_id][1])


    return denotation_to_idx


def json_to_pure(json_file: str, ):
    with open(json_file, encoding='utf-8') as f:
        json_dict = json.load(f)

    if json_dict.get('relations'):
        source_id = json_dict['sourceid']
        text = json_dict['text']
        denotation_list = json_dict[ 'denotations' ]
        relation_list = json_dict[ 'relations' ]
    else:
        return 'Noun'

    id_to_denotation = {}
    for denotation_dict in denotation_list:
        d_id = denotation_dict['id']
        begin = denotation_dict['span']['begin']
        end = denotation_dict['span']['end']
        obj = denotation_dict['obj']
        token = text[begin: end]
        id_to_denotation[d_id] = (token, obj, (begin, end))

    sent_to_offset, sent_list = get_sent_offset(text)

    denotation_to_sent, sent_to_denotation = denotation_sent_map(id_to_denotation, sent_to_offset)

    sent_to_label_pair = defaultdict(set)
    for relation_dist in relation_list:
        pred = relation_dist[ 'pred' ]
        subj = relation_dist[ 'subj' ]
        obj = relation_dist[ 'obj' ]
        subj_sent = denotation_to_sent[ subj ][ 0 ]
        obj_sent = denotation_to_sent[ obj ][ 0 ]
        # delete the relation denotation
        # if two token do not in the save sentence.
        if subj_sent != obj_sent:
            continue
        sent_to_label_pair[subj_sent].add((subj, obj, pred))

    document_dict = {"doc_key": source_id, "sentence": [],
                     "ner": [], "relations": []}
    for sent in sent_list:

        doc = nlp(sent)
        token_list = [ token.text for token in doc ]
        # fixme: delete the sentence without denotation
        if not sent_to_label_pair.get(sent):

            document_dict['sentence'].append(token_list)
            document_dict['ner'].append([])
            document_dict['relations'].append([])
            return document_dict
            # continue

        denotation_set = sent_to_denotation[sent]

        denotation_offset = {d_id:denotation_to_sent[d_id][1]
                             for d_id in denotation_set}

        token_offset = get_token_offset(sent, token_list)
        denotation_to_idx = denotation_token_map(token_offset, denotation_offset,
                                                 id_to_denotation)

        relation_list = []
        for (obj, sub, pred) in sent_to_label_pair[sent]:
            obj_sp, obj_ep, _ = denotation_to_idx[obj]
            sub_sp, sub_ep, _ = denotation_to_idx[sub]
            relation_list.append([obj_sp, obj_ep, sub_sp, sub_ep, pred])

        document_dict['sentence'].append(token_list)
        document_dict["ner"].append([denotation_to_idx[d_id] for d_id in sent_to_denotation[sent]])
        document_dict['relations'].append(relation_list)

    return document_dict


def batch_json_to_pure(json_file_path: str, save_file: str):

    json_file_list = os.listdir(json_file_path)
    wf = open(save_file, 'w', encoding='utf-8')
    for file_name in json_file_list:
        json_file = os.path.join(json_file_path, file_name)
        document_dict = json_to_pure(json_file)
        wf.write(f'{str(document_dict)}\n')
    wf.close()
    print(f'{save_file} save done.')


if __name__ == '__main__':
    # json_file = '../data/AGAC_training/json/PubMed-18594199.json'

    from src.config import config

    args = config()

    # document_dict = json_to_pure(json_file)
    batch_json_to_pure(args.train_json_path, args.pure_train_file)


    batch_json_to_pure(args.test_json_path, args.pure_test_file)
