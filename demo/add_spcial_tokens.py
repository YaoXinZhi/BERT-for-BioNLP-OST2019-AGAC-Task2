# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 12/04/2021 11:29
@Author: XINZHI YAO
"""

"""
update the vocab dictionary of tokenizer
https://huggingface.co/transformers/main_classes/tokenizer.html#
tokenizer.SPECIAL_TOKENS_ATTRIBUTES
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]
"""

import torch
from transformers  import BertModel, BertTokenizer

model_name = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)

vocab_dict = tokenizer.get_vocab()

# 未添加特殊Token之前
print(tokenizer.encode("<#> this <#>"))
print(tokenizer.encode("<s> the <#> this <#> a <$> body <$> end </s>"))

print("#" * 20)
special_tokens_dict = {'additional_special_tokens': [ "<#>", "<$>" ]}
# tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append("jin_token")
print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
# 添加特殊Token, 使模型不会拆分， 用作标记使用
tokenizer.add_special_tokens(special_tokens_dict)
print(tokenizer.additional_special_tokens)
print(tokenizer.additional_special_tokens_ids)
print(tokenizer.encode("<#> this <#>"))
print(tokenizer.encode("<s> the <#> this <#> a <$> body <$> end </s>"))

# resized the model's embedding matrix
roberta = RobertaModel.from_pretrained(pretrained_weights)
roberta.resize_token_embeddings(len(tokenizer))  # 调整嵌入矩阵的大小

