# -*- coding: utf-8 -*-
"""
Spyder Editor

set up for BERT
"""


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    import numpy as np
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    import datetime
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


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