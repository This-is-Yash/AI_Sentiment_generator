# train_sentiment.py
# Custom Sentiment Analysis Model Training Script

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1) Load dataset (SST-2 = positive/negative)
dataset = load_dataset("glue", "sst2")

# 2) Preprocess text
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(preprocess, batched=True)
encoded = encoded.remove_columns(["sentence", "idx"])

# 3) Define model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4) Metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="binary")
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 5) Training setup
training_args = TrainingArguments(
    output_dir="models/sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_steps=100,
)

# 6) Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

# 7) Save final model
trainer.save_model("models/sentiment")
tokenizer.save_pretrained("models/sentiment")

print("Model training complete! Saved to models/sentiment/")
