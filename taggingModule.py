#!/usr/bin/env python
# coding: utf-8

# In[28]:


from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification

class TaggingModule():
    
    def __init__(self, dataset_name, model_name, output_dir):
        self.datasets = load_dataset(dataset_name) #load dataset, should be ner data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #use pre trained tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer) #processes data (e.g pad)
        self.label_list = self.datasets["train"].features[f"{'ner'}_tags"].feature.names
        self.model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                                     num_labels=len(self.label_list))
        self.output_dir = output_dir #trained model will be here
        
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"{'ner'}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to  -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs                               
                                           
    def train_model(self):
        tokenized_datasets = self.datasets.map(tokenize_and_align_labels, batched=True)       
        args = TrainingArguments(
            self.output_dir,
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            num_train_epochs=1,
            weight_decay=0.01
        )
        trainer = Trainer(
            self.model,
            args,
            train_dataset = tokenized_datasets["train"],
            eval_dataset = tokenized_datasets["validation"],
            data_collator = self.data_collator,
            tokenizer = self.tokenizer
        )
        trainer.train()
    
module = TaggingModule("conll2003", "distilbert-base-cased", "taggingOutput")      
module.train_model()                                        


# In[ ]:




