import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Read the Excel files containing the data into dataframes
temp_df = pd.read_excel("train_data.xlsx")
test_df = pd.read_excel("test_data.xlsx")

# Split the training data into a training set and a validation set
train_df, val_df = train_test_split(
    temp_df, test_size=100, stratify=temp_df["label"], random_state=27
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")

# Tokenize data
train_encodings = tokenizer((train_df["text"].tolist())[:36], padding=True, truncation=True)
valid_encodings = tokenizer((val_df["text"].tolist())[:4], padding=True, truncation=True)
test_encodings = tokenizer((test_df["text"].tolist())[:20], padding=True, truncation=True)


# Prepare datasets
class ReadabilityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train = ReadabilityDataset(train_encodings, (train_df["label"].tolist())[:36])
validation = ReadabilityDataset(valid_encodings, (val_df["label"].tolist())[:4])
test = ReadabilityDataset(test_encodings, (test_df["label"].tolist())[:20])

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    "DTAI-KULeuven/robbert-2023-dutch-large", num_labels=4
)

# Specify evaluation metrics
accuracy = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")


# Define evaluation method
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(
            predictions=preds, references=labels, average="weighted"
        )["precision"],
        "recall": recall.compute(
            predictions=preds, references=labels, average="weighted"
        )["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")[
            "f1"
        ],
    }


# Define training arguments
args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=250,
    weight_decay=0.01,
    output_dir="./results",
    logging_dir="./logs",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=validation,
    compute_metrics=compute_metrics,
)

# Train, evaluate and save the model
trainer.train()
trainer.evaluate(test)
trainer.save_model("Neural_ARA_Model")
