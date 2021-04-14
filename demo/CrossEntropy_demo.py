# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 15:32
@Author: XINZHI YAO
"""

import torch
import torch.nn as nn


# fake data
# batch_size, label_size
x_input = torch.randn(3,3)
print(f'x_input: {x_input}')
# 1, label_size
y_target = torch.LongTensor([1,2,0])

# 计算输入softmax，此时可以看到每一行加到一起结果都是1
softmax_func = nn.Softmax(dim=1)
soft_output = softmax_func(x_input)
print(f'soft_output: {soft_output}')

predict_label = torch.argmax(soft_output, dim=-1)

# 1, 1, 1, 1

# 在softmax的基础上取log
log_output = torch.log(soft_output)
print(f'log_output: {log_output}')

# 对比softmax与log的结合与nn.Log-Softmax loss(负对数似然损失)的输出结果
# 发现两者是一致的。
logsoftmax_func=nn.LogSoftmax(dim=1)
logsoftmax_output=logsoftmax_func(x_input)
print('logsoftmax_output:\n',logsoftmax_output)


# pytorch中关于NLLLoss的默认参数配置为：
# reducetion=True、size_average=True
nllloss_func=nn.NLLLoss()
nlloss_output=nllloss_func(logsoftmax_output, y_target)
print('nlloss_output:\n',nlloss_output)


# 直接使用pytorch中的loss_func=nn.CrossEntropyLoss()
# 看与经过NLLLoss的计算是不是一样
crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
print('crossentropyloss_output:\n',crossentropyloss_output)

