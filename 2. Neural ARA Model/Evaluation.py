import torch
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from transformers import AutoTokenizer

# Load the saved model
model = AutoModelForSequenceClassification.from_pretrained("Neural_ARA_Model")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")

# Read the Excel files containing the data into dataframes
test_df = pd.read_excel("test_data.xlsx")

# Tokenize data
test_encodings = tokenizer((test_df["text"].tolist()), padding=True, truncation=True)

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

test = ReadabilityDataset(test_encodings, (test_df["label"].tolist()))

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
)

# Predict on the test set
predictions = trainer.predict(test)

# Get the predicted labels
pred_labels = predictions.predictions.argmax(-1)

# Calculate the confusion matrix
cm = confusion_matrix(test.labels, pred_labels)

# Calculate accuracy, precision, recall, and F1 score
report = classification_report(test.labels, pred_labels, output_dict=True)
accuracy = report['accuracy']
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

# Print the results
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
