import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split

file_path = "/training.1600000.processed.noemoticon.csv"
df = pd.read_csv(file_path, encoding='latin-1', header=None, usecols=[0, 5], names=['label', 'text'])

df['label'] = df['label'].map({0: 0, 4: 1})


train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(), "label": torch.tensor(self.labels[idx])}


train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = Adam(model.parameters(), lr=2e-5)

def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader)}")

train_model(model, train_loader, val_loader, optimizer)