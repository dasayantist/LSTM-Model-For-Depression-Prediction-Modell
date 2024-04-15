import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset Preparation
# Loading and preprocessing the labeled dataset
texts = ["Manchester United losing games every time will kill me one day"]
labels = [1] 

# Tokenization and Data Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text data using BertTokenizer
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]

# Pad sequences to ensure equal length
max_len = max(len(text) for text in tokenized_texts)
padded_texts = [text + [tokenizer.pad_token_id] * (max_len - len(text)) for text in tokenized_texts]

# Convert data into tensors
input_ids = torch.tensor(padded_texts)
labels = torch.tensor(labels)

# Model Configuration
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)

train_data = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

val_data = TensorDataset(val_inputs, val_labels)
val_loader = DataLoader(val_data, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

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
    
    model.eval()
    val_preds = []
    val_true_labels = []
    for batch in val_loader:
        batch_inputs, batch_labels = batch
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        
        with torch.no_grad():
            outputs = model(batch_inputs)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()
        val_preds.extend(preds)
        val_true_labels.extend(batch_labels.tolist())
    
    val_accuracy = accuracy_score(val_true_labels, val_preds)
    print(f"Epoch {epoch + 1}/{num_epochs}: Validation Accuracy: {val_accuracy:.4f}")

# Inference
def classify_text(text):
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    padded_text = tokenized_text + [tokenizer.pad_token_id] * (max_len - len(tokenized_text))
    input_ids = torch.tensor(padded_text).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits
    pred_label = torch.argmax(logits, dim=1).item()
    return pred_label

twitter_text = "Manchester United losing games every time will kill me one day"
prediction = classify_text(twitter_text)
print("Prediction:", prediction)