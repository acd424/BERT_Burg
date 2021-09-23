# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:59:39 2021

@author: mm18acd
"""


def prepare_data(sentences, labels, max_len, tokenizer):
    
    import torch
    from torch.utils.data import TensorDataset, random_split, Dataset

    print('Processing data')

    encodings = tokenizer(sentences,max_length = max_len, truncation = True, padding=True)


    
    
    class textDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    

    dataset = textDataset(encodings, labels)
    
    return dataset