
def trainer_api_call():
	import transformers
	from transformers import AutoModelForSequenceClassification
	from datasets import load_metric
	import numpy as np
	from transformers import TrainingArguments
	from transformers import AutoTokenizer
	from transformers import Trainer
	from prepare_datasets import prepare_data
	import pandas as pd


	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

	metric = load_metric(path = "./matthews_correlation/matthews_correlation.py")

	max_len = 450

	valdf = pd.read_csv('data/validation_set_2_recode.csv', engine = 'c')
	val_sentences = valdf.CrimeNotes.values.tolist()
	val_labels = valdf.force_used.values
	val_dataset = prepare_data(val_sentences, val_labels, max_len, tokenizer)


	## read in the training set #######
	training_set = 'data/force_data_for_label_updated_7.csv'
	df = pd.read_csv(training_set, engine = 'c')
	sentences = df.CrimeNotes.values.tolist()
	labels = df.force_used.values

	train_dataset = prepare_data(sentences, labels, max_len, tokenizer)



	def compute_metrics(eval_pred):
    
    
    		predictions, labels = eval_pred
    		predictions = np.argmax(predictions, axis = 1)
    		#must specify predictions and references
    		return metric.compute(predictions = predictions,references =  labels)




	training_args = TrainingArguments(
    	output_dir='./results',          # output directory
    	overwrite_output_dir = True,     
    	num_train_epochs=2,              # total number of training epochs
    	evaluation_strategy = "epoch",   # evaluate at end of epoch
    	learning_rate = 2e-5,                # learning rate for optimizer
    	adam_epsilon = 1e-8,                 # for optimizer
    	per_device_train_batch_size=16,  # batch size per device during training
    	per_device_eval_batch_size=64,   # batch size for evaluation
    	warmup_steps=500,                # number of warmup steps for learning rate scheduler
   	weight_decay=0.01,               # strength of weight decay
    	logging_dir='./logs',            # directory for storing logs
    	logging_strategy= "epoch"
	)


	trainer = Trainer(
    		model=model, 
    		args=training_args, 
    		train_dataset=train_dataset, 
    		eval_dataset=val_dataset,
    		compute_metrics = compute_metrics
		)


	trainer.train()