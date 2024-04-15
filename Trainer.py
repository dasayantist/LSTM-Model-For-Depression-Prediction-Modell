import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

# Loading text from CSV file
df = pd.read_csv('Mental-Health-Twitter/Mental-Health-Twitter.csv') 
texts = df['post_text'].tolist()
labels = df['label'].tolist() 

# Tokenization and Data Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
max_len = max(len(text) for text in tokenized_texts)
padded_texts = [text + [tokenizer.pad_token_id] * (max_len - len(text)) for text in tokenized_texts]
input_ids = torch.tensor(padded_texts)
labels = torch.tensor(labels)

# Model Configuration
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training
train_data = TensorDataset(input_ids, labels)
train_loader = DataLoader(train_data, batch_size=1)  # Use batch_size=1 for small datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, no_deprecation_warning=True)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        batch_inputs, batch_labels = batch
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
val_preds = []
val_true_labels = []
for batch in train_loader:
    batch_inputs, batch_labels = batch
    batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
    
    with torch.no_grad():
        outputs = model(batch_inputs)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).tolist()
    val_preds.extend(preds)
    val_true_labels.extend(batch_labels.tolist())

val_accuracy = accuracy_score(val_true_labels, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")
