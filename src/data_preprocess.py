# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 12/04/2021 11:05
@Author: XINZHI YAO
"""

import os
import json
import logging
from nltk import tokenize
from collections import defaultdict
from itertools import product, combinations

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)

global write_pair_to_neg


def get_sent_offset(text: str):
    sent_list = tokenize.sent_tokenize(text)
    begin = 0
    sent_to_offset = {}
    sent_to_id = {}
    for sent in sent_list:
        end = begin + len(sent)
        sent_to_offset[sent] = (begin, end)
        sent_to_id[sent] = len(sent_to_id)
        begin = end + 1
    return sent_to_offset, sent_to_id


def denotation_sent_map(id_to_denotation: dict, sent_to_offset: dict):

    denotation_to_sent = {}

    for d_id, (token, label, (t_begin, t_end)) in id_to_denotation.items():
        for sent, (s_begin, s_end) in sent_to_offset.items():

            if t_begin >= s_begin:
                if t_end <= s_end:

                    t_begin -= s_begin
                    t_end = t_begin + len(token)

                    offset = (t_begin, t_end)
                    if sent[t_begin: t_end] != token:
                        if sent.count(token) == 1:
                            offset = (sent.find(token), sent.find(token)+len(token))
                        else:
                            print(sent)
                            print(t_begin, t_end)
                            print(sent.find(token), sent.find(token)+len(token))
                            print(token, sent[t_begin: t_end])

                    denotation_to_sent[d_id] = (sent, offset)

                    break
                else:
                    pass
    return denotation_to_sent


def neg_data_sampler(denotation_to_sent: dict, id_to_denotation: dict,
                     sent_to_label_pair: dict,
                     sent_to_offset: dict, pair_to_count: dict):

    label_pair_set = set(pair_to_count.keys())

    hard_neg_data_list = []
    normal_neg_data_list = []

    sent_to_denotation = defaultdict(set)
    for denotation, (sent, _) in denotation_to_sent.items():
        sent_to_denotation[sent].add(denotation)

    for sent, denotation_set in sent_to_denotation.items():
        sent_begin, sent_end = sent_to_offset[sent]
        pair_set = set(combinations(denotation_set, 2))
        for (subj, obj) in pair_set:
            if (subj, obj) in sent_to_label_pair[sent] or (obj, subj) in sent_to_label_pair[sent]:
                continue

            subj_token, subj_label, (subj_begin, subj_end) = id_to_denotation[subj]
            obj_token, obj_label, (obj_begin, obj_end) = id_to_denotation[obj]

            subj_begin -= sent_begin
            subj_end = subj_begin + len(subj_token)

            obj_begin -= sent_begin
            obj_end = obj_begin + len(obj_token)

            if ((subj_label, obj_label) in label_pair_set) and\
                    (write_pair_to_neg[(subj_label, obj_label)] < pair_to_count[(subj_label, obj_label)]):
                write_pair_to_neg[(subj_label, obj_label)] += 1
                hard_neg_data_list.append(list(map(str, (subj_token, subj_label, (subj_begin, subj_end),
                                                                 obj_token, obj_label, (obj_begin, obj_end),
                                                                 'NoRelation', sent))))
            elif ((obj_label, subj_label) in label_pair_set) and \
                    (write_pair_to_neg[(obj_label, subj_label)] < pair_to_count[(obj_label, subj_label)]):
                write_pair_to_neg[(obj_label, subj_label)] += 1
                hard_neg_data_list.append(list(map(str, (obj_token, obj_label, (obj_begin, obj_end),
                                                             subj_token, subj_label, (subj_begin, subj_end),
                                                             'NoRelation', sent))))
            else:
                normal_neg_data_list.append(list(map(str, (subj_token, subj_label, (subj_begin, subj_end),
                                                                 obj_token, obj_label, (obj_begin, obj_end),
                                                                 'NoRelation', sent))))

    return hard_neg_data_list, normal_neg_data_list



def json_to_text(json_file: str, pair_to_count: dict):


    with open(json_file) as f:
        json_dict = json.load(f)

    if json_dict.get('relations'):

        text = json_dict['text']
        denotation_list = json_dict['denotations']
        relation_list = json_dict['relations']
    else:
        return 'Noun'

    sent_to_offset, sent_to_id = get_sent_offset(text)

    id_to_denotation = {}
    for denotation_dict in denotation_list:
        d_id = denotation_dict['id']
        begin = denotation_dict['span']['begin']
        end = denotation_dict['span']['end']
        obj = denotation_dict['obj']

        token = text[begin: end]

        id_to_denotation[d_id] = (token, obj, (begin, end))

    denotation_to_sent = denotation_sent_map(id_to_denotation, sent_to_offset)

    for t_id, (sentence, (t_start, t_end)) in denotation_to_sent.items():
        token = id_to_denotation[t_id][0]
        token_in_sent = sentence[t_start: t_end]
        if token != token_in_sent:
            print(json_file)
            print(t_id)
            print(sentence)
            print((t_start, t_end))
            print(token, token_in_sent)

    pos_data_list = []
    sent_to_label_pair = defaultdict(set)
    for relation_dist in relation_list:
        # r_id = relation_dist['id']
        pred = relation_dist['pred']
        subj = relation_dist['subj']
        obj = relation_dist['obj']

        subj_token, subj_label, _ = id_to_denotation[subj]
        subj_sent, subj_offset = denotation_to_sent[subj]

        obj_token, obj_label, _ = id_to_denotation[obj]
        obj_sent, obj_offset = denotation_to_sent[obj]

        # delete the relation denotation if two token do not in the save sentence.
        if subj_sent != obj_sent:
            continue

        pos_data_list.append(list(map(str, [subj_token, subj_label, subj_offset,
                                            obj_token, obj_label, obj_offset,
                                            pred, subj_sent])))


        sent_to_label_pair[subj_sent].add((subj, obj))


    hard_neg_data_list, normal_neg_data_list = neg_data_sampler(denotation_to_sent, id_to_denotation,
                                   sent_to_label_pair, sent_to_offset,
                                   pair_to_count)
    return pos_data_list, hard_neg_data_list, normal_neg_data_list


def read_relation_statistics(relation_statistics_file: str):

    pair_to_count = defaultdict(int)
    with open(relation_statistics_file) as f:
        f.readline()
        for line in f:
            label_1, _, label_2, count = line.strip().split('\t')
            pair_to_count[(label_1, label_2)] += int(count)
    return pair_to_count


def batch_json_to_text(json_file_path: str, save_file: str,
                       train_relation_statistics: str, save_normal_sample=True):


    pos_data_count = 0
    hard_neg_data_count = 0
    normal_neg_data_count = 0

    normal_neg_data_list = []

    label_pair_to_count = read_relation_statistics(train_relation_statistics)

    json_file_list = os.listdir(json_file_path)
    wf = open(save_file, 'w', encoding='utf-8')
    wf.write('Token1\tLabel1\tOffset1\t'
             'Token2\tLabel2\tOffset2\t'
             'Relation\tSentence\n')
    for file_name in json_file_list:
        json_file = os.path.join(json_file_path, file_name)
        # break
        pos_data_list, neg_data_list, normal_neg = json_to_text(json_file, label_pair_to_count)
        for data in normal_neg:
            normal_neg_data_list.append(data)

        pos_data_count += len(pos_data_list)
        for pos_data in pos_data_list:
            write_line = '\t'.join(pos_data)
            if '\n' in write_line:
                print(pos_data[-1])
                print()
            wf.write(f'{write_line}\n')

        hard_neg_data_count += len(neg_data_list)
        for neg_data in neg_data_list:
            write_line = '\t'.join(list(map(str, neg_data)))
            wf.write(f'{write_line}\n')

    if save_normal_sample:
        normal_neg_data_list = normal_neg_data_list[:pos_data_count - hard_neg_data_count]
        # print(len(normal_neg_data_list),
        #       pos_data_count-hard_neg_data_count
        #       ,pos_data_count, hard_neg_data_count)
        for data in normal_neg_data_list:
            write_line = '\t'.join(list(map(str, data)))
            wf.write(f'{write_line}\n')
            normal_neg_data_count += 1


    # print(write_pair_to_neg)
    print(f'Positive data: {pos_data_count:,}, Negative(NoRelation) data: {hard_neg_data_count + normal_neg_data_count:,}'
          f'(Hard: {hard_neg_data_count:,},'
          f' Normal: {normal_neg_data_count:,}).')
    print(f'{save_file} save done.')
    wf.close()



if __name__ == '__main__':
    # from src.config import config

    args = config()

    """
    Train data:
    Positive data: 2,614, Negative data: 2,614(Hard: 1,317, Normal: 1,297).
    Test data:
    Positive data: 1,873, Negative data: 1,873(Hard: 550, Normal: 1,323).
    
    ThemeOf : CauseOf : NoRelation should be 1:1:1
    """

    print('Train data:')
    write_pair_to_neg = defaultdict(int)
    batch_json_to_text(args.train_json_path, args.train_file,
                       args.train_relation_statistics)

    print('Test data:')
    write_pair_to_neg = defaultdict(int)
    batch_json_to_text(args.test_json_path, args.test_file, args.test_relation_statistics)
