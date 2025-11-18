from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
import json
from torch.utils.data import Dataset  # 👈 新增：自定义 Dataset

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 加载数据
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

# 标签映射（提到外面避免重复）
label_to_id = {"拍照": 0, "取消": 1, "购买": 2, "其他": 3}

# 👇 自定义 Dataset 类（关键修复！）
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = self.label_map[item["label"]]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        print(encoding)
        # 注意：squeeze(0) 去掉 batch 维度，因为 __getitem__ 返回单样本
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 创建 Dataset 实例
train_dataset = IntentDataset(train_data, tokenizer, label_to_id)
val_dataset = IntentDataset(val_data, tokenizer, label_to_id)

# 加载模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=4
)

# 训练配置
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,      # 小 batch 防 OOM
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 开始训练
trainer.train()
tokenizer.save_pretrained("./results/final_model")