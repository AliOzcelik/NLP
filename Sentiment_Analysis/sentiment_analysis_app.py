import streamlit as st
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertTokenizer, get_linear_schedule_with_warmup


# Define your model architecture
class TextClassifier(nn.Module):
    
    def __init__(self, bert_model_name, num_classes):
        super(TextClassifier, self).__init__()
        # self.bert = DistilBertForSequenceClassification.from_pretrained(bert_model_name)
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        #print(self.bert.config)
        outputs = self.bert(input_ids, attention_mask)
        # pooler_outputs = outputs.pooler_output
        hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        pooler_outputs = hidden_state[:, 0] 
        x = self.dropout(pooler_outputs)
        x = self.fc(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

bert_model_name = "distilbert-base-uncased"
model = TextClassifier(bert_model_name, num_classes=6)
model.load_state_dict(torch.load('sentiment_model.pth'))
model = model.to(device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)


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


# Streamlit UI
st.title("Sentiment Analysis with DistilBERT")
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
       
        
        predicted_class = predict_sentiment(user_input, model, tokenizer, device, max_length=128)

        st.success(f"Predicted class: {predicted_class}")




 # Tokenize input
 # inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
 # input_ids = inputs["input_ids"]
 # attention_mask = inputs["attention_mask"]

 # # Predict
 # with torch.no_grad():
 #     outputs = model(input_ids, attention_mask)
 #     predicted_class = torch.argmax(outputs, dim=1).item()
 