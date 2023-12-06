import pandas as pd
import gzip
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
import numpy as np

file_path = './Home_and_Kitchen_5.json.gz'

with gzip.open(file_path) as f:
  df = pd.read_json(f, lines = True)

df= df[['overall','verified','reviewTime','reviewText']]
df= df[df['verified'] == True].reset_index(drop=True)
df.dropna(subset=['reviewText', 'overall'], inplace=True)
df['overall'] = df['overall'] - 1  # Adjust ratings to start from 0

df_train,df_test= train_test_split(df,test_size=0.2)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# df_test = df.iloc[:300, :].reset_index(drop=True)
# df_train = df.iloc[300:, :].reset_index(drop=True)

train_dataset = Dataset.from_pandas(df_train,)
test_dataset = Dataset.from_pandas(df_test)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    tokenized_inputs = tokenizer(batch['reviewText'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    tokenized_inputs["labels"] = torch.tensor(batch['overall'])
    return tokenized_inputs

train_dataset = Dataset.from_pandas(df_train).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(df_test).map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(np.unique(df['overall']))
)

training_args = TrainingArguments(
    output_dir='./results_home',
    num_train_epochs=1,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    fp16=True
)

# Function to compute metrics
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, roc_auc_score, accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    # Hard predictions are needed for accuracy, precision, recall, and F1
    hard_preds = np.argmax(preds, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, hard_preds, average='weighted')
    acc = accuracy_score(labels, hard_preds)
    mae = mean_absolute_error(labels, hard_preds)

    # Compute ROC AUC for each class
    roc_auc = {}
    for i in range(preds.shape[1]):  # Iterate over each class
        roc_auc[f"roc_auc_class_{i}"] = roc_auc_score((labels == i).astype(int), preds[:, i])

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mae': mae,
        **roc_auc  # This will expand the dictionary to include the roc_auc for each class
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()