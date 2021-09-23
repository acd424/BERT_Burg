"""
Created on Sat Feb 13 15:26:14 2021

@author: mm18acd
"""


import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import transformers
import torch
import time
import datetime
import random

from additional_functions import get_max_len, format_time, flat_accuracy
from prepare_data import prepare_data
from training_function import training_function
from label_unlabelled import label_unlabelled

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig




#######################  Main training/model variables ############################## 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)




# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

#### CPU or CUDA
device = torch.device("cpu")


# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4



# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 16


#### Find the max token length of the individual texts 

# find max length from all documents
df = pd.read_csv('data/recoded_data_for_test_3.csv', engine = 'c')

sentences = df.CrimeNotes.values
labels = df.motorvehicle.values

max_len = get_max_len(sentences, tokenizer)



### read in the validation set 
valdf = pd.read_csv('data/validation_set_2_recode.csv', engine = 'c')
val_sentences = valdf.CrimeNotes.values
val_labels = valdf.motorvehicle.values
val_dataset = prepare_data(val_sentences, val_labels, max_len, tokenizer)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


######### Model build n ########

###set the correct variables
n = 1
training_set = 'data/recoded_data_for_label.csv'


## read in the training set #######
df = pd.read_csv(training_set, engine = 'c')
sentences = df.CrimeNotes.values
labels = df.motorvehicle.values

train_dataset = prepare_data(sentences, labels, max_len, tokenizer)


# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )


description = "Burg data, car model " +str(n) # description appended to log files 
#### run the model
model = training_function(epochs, model, train_dataloader, validation_dataloader, optimizer, device, training_set, description)
# updates epoc_metrics.txt

model_name = "model.BERT.burg.car.run_" +str(n) +".pth"

torch.save(model, model_name)


###### label all of the data and the validation set


### these functions ingest the data from the first filename and then creates a labelled set with probabilites with the second filename

save_results_to = 'data/results_val_' +str(n) + '.csv'

label_unlabelled(test_df_filename = 'data/validation_set_2_recode.csv', 
                 out_filename = save_results_to,
                 text_col = 'CrimeNotes',
                 label_col = 'motorvehicle',
                 max_len = max_len, 
                 tokenizer = tokenizer, 
                 model = model, 
                 metrics = 'TRUE', 
                 description = description)