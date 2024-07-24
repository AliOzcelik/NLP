import numpy as np
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

# to_map_style_dataset: Converts iterable-style dataset to map-style dataset.


train_iter = AG_NEWS(split="train")
test_iter = AG_NEWS(split="test")

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

#device = torch.device('mps')
device = torch.device('cpu')


def collate_batch(batch):
    
    label_list, text_list, offsets = [], [], [0]
    for label, text in batch:
        label_list.append(int(label)-1)
        processed_text = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
        
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    
    return label_list.to(device), text_list.to(device), offsets.to(device)



train_dataLoader = DataLoader(to_map_style_dataset(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataLoader = DataLoader(to_map_style_dataset(test_iter), batch_size=64, shuffle=False, collate_fn=collate_batch)


class TextClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, num_layers, hidden_size, num_class):
        super(TextClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        #self.fc = nn.Linear(hidden_size, num_class)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_class)
        self.dropout = nn.Dropout(0.2)
        self.init_weights()
            
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        
    def forward(self, text, offsets, h):
        x = self.embedding(text, offsets)
        lstm_out, (h, c) = self.lstm(x, h)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.fc2(lstm_out)
        return lstm_out
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, self.hidden_size)).to(device)
        c0 = torch.zeros((self.num_layers, self.hidden_size)).to(device)
        hidden = (h0,c0)
        return hidden


num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
embedding_size = 64
num_layers = 1
hidden_size = 256
batch_size=64

model = TextClassifier(vocab_size, embedding_size, num_layers, hidden_size, num_class).to(device)
epochs = 10  
learning_rate = 0.001  

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

def train(dataloader):
    
    model.train()
    h_init = model.init_hidden(batch_size)
    
    for idx, (label, text, offsets) in enumerate(dataloader):
        h = tuple([each.data for each in h_init])
        
        optimizer.zero_grad()
        predicted_label = model.forward(text, offsets, h)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()


def evaluate(dataloader):
    
    model.eval()
    total_acc, total_count = 0, 0
    val_h_init = model.init_hidden(batch_size)
    
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            
            label = label.to(device)
            text = text.to(device)
            #offsets = offsets.to(device)
            val_h = tuple([each.data for each in val_h_init])
            
            predicted_label = model(text, offsets, val_h)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            
    return total_acc / total_count, loss

total_accuracy = None
for epoch in range(epochs):
    tic = time.time()
    train(train_dataLoader)
    accuracy, loss = evaluate(test_dataLoader)
    if total_accuracy is not None and total_accuracy > accuracy:
        scheduler.step()
    else:
        total_accuracy = accuracy
        
    print("------------------")
    print("epoch: ", epoch+1)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    toc = time.time()
    print("epoch time: ", toc-tic)


ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}


def predict(text):
    with torch.no_grad():
        text = torch.tensor(vocab(tokenizer(text)))
        h = tuple([each.data for each in model.init_hidden(batch_size)])
        output = model(text, torch.tensor([0]), h)
        return output.argmax(1).item() + 1


ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

result = predict(ex_text_str)

print("This is a {} news".format(ag_news_label(result)))