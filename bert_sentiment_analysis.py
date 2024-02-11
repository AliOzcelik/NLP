import numpy as np
import pandas as pd
import transformers as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time
import re
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



training = open("/Users/desidero/Desktop/NLP/Emotions/train.txt","r",encoding="utf8", errors="ignore").read().split("\n")
val_text = open("/Users/desidero/Desktop/NLP/Emotions/val.txt","r",encoding="utf8", errors="ignore").read().split("\n")
test_text = open("/Users/desidero/Desktop/NLP/Emotions/test.txt","r",encoding="utf8", errors="ignore").read().split("\n")

train_text = training + val_text

class Preprocess():
    
    def __init__(self, train_text, test_text, tokenizer):
        self.tokenizer = tokenizer
        self.train_text, self.train_label = self.split(train_text) 
        self.test_text, self.test_label = self.split(test_text)
        self.num_classes = self.find_num_classes()
        self.map_emotions()
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text) # replace "i'm" with "i am"
        text = re.sub(r"im", "i am", text)
        text = re.sub(r"ive", "i have", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"n't", "not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"wont", "will not", text)
        text = re.sub(r"won t", "will not", text)
        text = re.sub(r"didn't", "did not", text)
        text = re.sub(r"didnt", "did not", text)
        text = re.sub(r"didn t", "did not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"cant", "cannot", text)
        text = re.sub(r"can t", "cannot", text)
        text = re.sub(r"[-()\"#/@:<>{}+=~|.?,!]", "", text)
        return text 
    
    def split(self, text):
        X = []
        y = []
        for i in text:
            if ";" not in i:
                pass
            else:
                ali = i.split(";")
                X.append(self.clean_text(ali[0]))
                y.append(self.clean_text(ali[1]))
        return X, y
    
    def map_emotions(self):
        emotions_dict = {"love":0, "sadness":1, "anger":2, "surprise":3, "joy":4, "fear":5}
        self.train_label = [emotions_dict[i] for i in self.train_label]
        self.test_label = [emotions_dict[i] for i in self.test_label]
    
    def find_num_classes(self):
        emotions = set([x for x in self.train_label])
        return  len(emotions)
    
    

class TextClassificationDataset():
    
    def __init__(self, text, labels, tokenizer, max_length):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    


max_length = 128
batch_size = 16
num_epochs = 10
learning_rate = 2e-5    

bert_model_name = "bert-base-uncased"
tokenizer = T.BertTokenizer.from_pretrained(bert_model_name)
device = torch.device('mps')

pre = Preprocess(train_text, test_text, tokenizer)
num_classes = pre.num_classes

train_dataset = TextClassificationDataset(pre.train_text, pre.train_label, tokenizer, max_length)
val_dataset = TextClassificationDataset(pre.test_text, pre.test_label, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)



class TextClassifier(nn.Module):
    
    def __init__(self, bert_model_name, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = T.BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooler_outputs = outputs.pooler_output
        x = self.dropout(pooler_outputs)
        x = self.fc(x)
        return x


model = TextClassifier(bert_model_name, num_classes).to(device)


def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        #loss = nn.BCELoss(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()



def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        
    emotions_dict = {0:"love", 1:"sadness", 2:"anger", 3:"surprise", 4:"joy", 5:"fear"}
    #return "positive" if preds.item() == 1 else "negative"
    return emotions_dict[preds.item()]


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = T.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


for epoch in range(num_epochs):
    tic = time.time()
    print(f"Epoch {epoch + 1}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
    toc = time.time()
    print("epoch time: ", toc-tic)



test_text_1 = "i am feeling like i can win every game"
sentiment = predict_sentiment(test_text_1, model, tokenizer, device)
print(test_text_1)
print(f"Predicted sentiment: {sentiment}")


test_text_2 = "i am feeling depressed"
sentiment = predict_sentiment(test_text_2, model, tokenizer, device)
print(test_text_2)
print(f"Predicted sentiment: {sentiment}")


test_text_3 = "i feel like butterflies are flying in my stomach"
sentiment = predict_sentiment(test_text_3, model, tokenizer, device)
print(test_text_3)
print(f"Predicted sentiment: {sentiment}")


test_text_4 = "i felt a fire in my head"
sentiment = predict_sentiment(test_text_4, model, tokenizer, device)
print(test_text_4)
print(f"Predicted sentiment: {sentiment}")

