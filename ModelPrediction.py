import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Loading pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

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

# Getting input text from user
user_text = input("Enter the text to classify: ")

# Classifying the user input
prediction, probabilities = classify_depression(user_text)

# Print the result
print("Prediction:", labels[prediction])
print("Probabilities:", probabilities)
