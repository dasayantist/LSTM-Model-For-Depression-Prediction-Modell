import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

# Load and preprocess your labeled dataset
df = pd.read_csv('your_file.csv')  # Replace 'your_file.csv' with the path to your CSV file
texts = df['text'].tolist()
labels = df['label'].tolist()  # Assuming 'label' is the column containing the labels

# Tokenization and Data Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
max_len = max(len(text) for text in tokenized_texts)
padded_texts = [text + [tokenizer.pad_token_id] * (max_len - len(text)) for text in tokenized_texts]
input_ids = torch.tensor(padded_texts)

# Create attention masks
attention_masks = [[1] * len(text) + [0] * (max_len - len(text)) for text in tokenized_texts]
attention_masks = torch.tensor(attention_masks)

# Convert labels to tensor
labels = torch.tensor(labels)

# Model Configuration
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training
train_data = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(train_data, batch_size=1)  # Use batch_size=1 for small datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        batch_inputs, batch_masks, batch_labels = tuple(t.to(device) for t in batch)
        
        optimizer.zero_grad()
        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
val_preds = []
val_true_labels = []
for batch in train_loader:
    batch_inputs, batch_masks, batch_labels = tuple(t.to(device) for t in batch)
    
    with torch.no_grad():
        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).tolist()
    val_preds.extend(preds)
    val_true_labels.extend(batch_labels.tolist())

val_accuracy = accuracy_score(val_true_labels, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")
