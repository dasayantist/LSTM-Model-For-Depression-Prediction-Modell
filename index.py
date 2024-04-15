import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # Assuming binary classification (depression or not)

# Define labels
labels = {0: "No depression symptoms", 1: "Depression symptoms"}

# Function to classify text
def classify_depression(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    outputs = model(**inputs)

    # Get predicted probabilities
    probs = torch.softmax(outputs.logits, dim=-1)

    # Get predicted label
    pred_label = torch.argmax(probs, dim=-1).item()

    return pred_label, probs.squeeze().detach().numpy()

# Example text
twitter_text = "Grab a banana for breakfast! They are known as a happy fruit. Eating just one can help relieve irritable emotions, anger and or depression."

# Classify the text
prediction, probabilities = classify_depression(twitter_text)

# Print the result
print("Prediction:", labels[prediction])
print("Probabilities:", probabilities)
