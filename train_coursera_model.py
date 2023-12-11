import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np

file_path = 'reviews.csv'

df = pd.read_csv(file_path)

df = df[['Review', 'Label']]
df['Label'] = df['Label'] - 1  # Adjust labels to start from 0

df_train, df_test = train_test_split(df, test_size=0.2)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Update tokenization function
def tokenize(batch):
    tokenized_inputs = tokenizer(batch['Review'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    tokenized_inputs["labels"] = torch.tensor(batch['Label'])
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(np.unique(df['Label']))  
).to("cuda")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_coursera_reviews',
    num_train_epochs=1,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,  
    save_steps=500,
    load_best_model_at_end=True
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
