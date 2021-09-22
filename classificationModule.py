#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class ClassificationModule():
    
    class MyDataset(torch.utils.data.Dataset): #nested class for dataset building
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor([self.labels[idx]])
            return item

        def __len__(self):
            return len(self.labels)

    def __init__(self, file_name, model_name, output_dir):
        self.file_name = file_name
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True,
                                                           padding=True, truncation=True,
                                                           max_length=512) #use pre trained tokenizer, model
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.output_dir = output_dir #trained model will be here
        
    def prepare_data(self):
        df = pd.read_json(self.file_name, lines=True) #read data
        df = df.dropna(axis=0, subset=['reviewText']) 
        df['text_word_count'] = df['reviewText'].apply(
            lambda x: len(str(x).split())) #count number of words in each sentence
        df['labels'] = np.random.randint(0, 2, df.shape[0]) #create artificial labels for classification                                  
        df = df.loc[df.text_word_count <= 300] 
        df = df.sample(n=1000)
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(df.reviewText.to_list(),
                                                                                df.labels.to_list(),
                                                                                test_size=0.3)
        train_encodings = self.tokenizer(train_texts, padding="max_length", truncation=True)
        valid_encodings = self.tokenizer(valid_texts, padding="max_length", truncation=True)
        train_dataset = MyDataset(train_encodings, train_labels)
        valid_dataset = MyDataset(valid_encodings, valid_labels)
        return train_dataset, valid_dataset
            
    def train_model(self, train_dataset, eval_dataset):    
        training_args = TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=50,
            evaluation_strategy="steps"
        )

        trainer = Trainer(model = self.model, args=training_args,
                          train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
    
module = ClassificationModule("Magazine_Subscriptions.json", "bert-base-uncased", "classificationOutput") 
train_data, validation_data = module.prepare_data()
module.train_model(train_data, validation_data)


# In[ ]:




