# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 13:43:01 2021

@author: mm18acd
"""

def get_max_len(sentences,tokenizer):

    max_len = 0

    # For every sentence...
    for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)

    return max_len