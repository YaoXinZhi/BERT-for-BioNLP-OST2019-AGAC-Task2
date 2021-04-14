# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 20:10
@Author: XINZHI YAO
"""
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




"""
About Micro, Macro, Weighted

Macro-average，Micro-average和Weighted-average就是三种汇聚所有类的指标的方式。
Marco: 
这相当于把所有类别的权重都是设置为一致，
这种方式在测试样本的类别在数量上极端不均衡的时候极端的不合理。
Micro:
    equal
Weighted: unbalance
    报告中weighted avg :对每一类别的f1_score进行加权平均，权重为各类别数在y_true中所占比例
    different label, different weight
"""

actual = [1, 2, 1, 2, 0,1,1]
# actual = [[1, 1, 1], [1, 0,1,0]]
predict = [2, 0, 2, 2, 1,1,1]
# predict = [[1, 1, 1], [1, 1,1,1]]

warnings.filterwarnings("ignore")
acc = accuracy_score(actual, predict, )
precision = precision_score(actual, predict, average='macro')
recall = recall_score(actual, predict, average='macro')
f1 = f1_score(actual, predict, average='macro')


