#!/usr/bin/env python
# fine_tune_taiwan_restaurant_intents.py
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import numpy as np
import torch
import pandas as pd
from collections import Counter
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
num_labels = 8

def load_excel_data(file_path):
    # 读取训练集和测试集
    train_df = pd.read_excel(file_path, sheet_name="训练集")
    test_df = pd.read_excel(file_path, sheet_name="测试集")
    
    # 重命名列名以匹配原代码
    train_df = train_df.rename(columns={"input": "text", "分类": "label"})
    test_df = test_df.rename(columns={"input": "text", "分类": "label"})
    
    # 创建标签字典
    label_names = train_df["label"].unique().tolist()
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # 将标签转换为数字
    train_df["label"] = train_df["label"].map(label2id)
    test_df["label"] = test_df["label"].map(label2id)
    
    # 转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset, label2id, id2label

# 加载数据
train_dataset, test_dataset, label2id, id2label = load_excel_data("../data/划分完数据集.xlsx")
dataset = {
    "train": train_dataset,
    "test": test_dataset
}
print("load data over ")


# Choose a pretrained model checkpoint.
model_checkpoint = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("load model over ")
# 3. 分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

# 分词处理
tokenized_datasets = {
    "train": dataset["train"].map(tokenize_function, batched=True),
    "test": dataset["test"].map(tokenize_function, batched=True)
}

# 4. 计算类别权重（处理不平衡数据）
label_list = tokenized_datasets["train"]["label"]
label_freq = Counter(label_list)
total = len(label_list)
class_weights = [total / label_freq[i] for i in range(len(label_freq))]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:", class_weights)

# Number of intent labels



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

print("init model over")
# -----------  Metrics (precision, recall, F1, accuracy) -----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # 处理logits和labels的形状
    if labels.ndim > 1:
        labels = labels.reshape(-1)  # 展平为1D数组
    
    # 获取预测结果
    if logits.shape[1] == 1:  # 二分类情况
        predictions = (logits > 0).astype(int).flatten()
    else:  # 多分类情况
        predictions = np.argmax(logits, axis=-1)
        if predictions.ndim > 1:
            predictions = predictions.reshape(-1)
    
    # 确保类型正确
    predictions = predictions.astype(int)
    labels = labels.astype(int)
    
    # 计算所有指标（使用sklearn，完全本地计算）
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro",zero_division=0),
        "recall": recall_score(labels, predictions, average="macro",zero_division=0),
        "f1": f1_score(labels, predictions, average="macro",zero_division=0)
    }

# -----------  Training arguments -----------
training_args = TrainingArguments(
    output_dir="./model/results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_dir='./model/logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True, 
    dataloader_pin_memory=False, 
)
print("TrainingArguments done,init trainer")

# -----------  Trainer -----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("training.........")
# Train!
trainer.train()

eval_results = trainer.evaluate(tokenized_datasets["test"])
print("\nFinal Evaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")

# Save final model
model.save_pretrained("./albert-model/final_model/")
tokenizer.save_pretrained("./albert-model/./final_model/")
