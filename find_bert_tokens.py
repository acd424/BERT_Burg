# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:46:50 2021

@author: mm18acd
"""

def find_BERT_tokens(tokenizer, filename):
    
    #filename = 'data\data_for_test.csv'
    import pandas as pd
    
    df = pd.read_csv(filename, engine = 'python')
    
    sentences = df.CrimeNotes.values
    #sen = sentences[0]
    
    token_ids = []
    tokens = []
    
    for sent in sentences:
        encoded = tokenizer.tokenize(sent)
        token_id = tokenizer.convert_tokens_to_ids(encoded)
        
        tokens.append(encoded)
        token_ids.append(token_id)
    

    flat_tokens = [item for sublist in tokens for item in sublist]
    flat_ids = [item for sublist in token_ids for item in sublist]
    
    tokens_df = pd.DataFrame(
    {'tokens': flat_tokens,
     'token_id': flat_ids
    })
    
    tokens_df.to_csv('BERT_tokens.csv')
    
