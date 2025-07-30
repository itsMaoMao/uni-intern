#!/usr/bin/env python
# fine_tune_taiwan_restaurant_intents.py

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import evaluate
import numpy as np
import torch
from collections import Counter


# Load the dataset from Hugging Face Hub
dataset = load_dataset("Luigi/dinercall-intent")

# Choose a pretrained model checkpoint.
model_checkpoint = "ckiplab/albert-base-chinese"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

# Tokenize
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Number of intent labels
num_labels = tokenized_datasets["train"].features["label"].num_classes

# ----------- 🛡️ Compute class weights for imbalanced training set -----------
label_list = tokenized_datasets["train"]["label"]
label_freq = Counter(label_list)
total = len(label_list)

# Inverse frequency as weight
class_weights = [total / label_freq[i] for i in range(num_labels)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:", class_weights)

# ----------- 🧠 Load model with weighted loss function -----------
from torch import nn
from transformers import BertPreTrainedModel, BertModel

class CustomModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# Initialize model
model = CustomModel.from_pretrained(model_checkpoint, num_labels=num_labels)

# ----------- 📊 Metrics (precision, recall, F1, accuracy) -----------
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    }

# ----------- ⚙️ Training arguments -----------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=True,# 👈 This will push your model after training
    hub_model_id="Luigi/albert-base-chinese-dinercall-intent",  # optional, use if you want a custom name
    hub_private_repo=False  # optional, if you want a private repo
)

# ----------- 🧪 Trainer -----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

# Save final model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
