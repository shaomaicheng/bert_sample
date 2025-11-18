import os

import torch
import json

from transformers import AutoTokenizer, BertForSequenceClassification


def predict_intent(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    # 得到概率列表 [p0, p1, p2, p3]
    probabilities = (torch.softmax(logits, dim=-1).squeeze().tolist())

    # 定义标签顺序（必须和训练时 label_to_id 一致！）
    labels = ["拍照", "取消", "购买", "其他"]

    # 将每个概率分别 round，并组成字典
    result = {
        label: round(prob, 4)
        for label, prob in zip(labels, probabilities)
    }
    return result


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained(
    "results/checkpoint-3",
    num_labels=4
)
# 使用示例
model.eval()
sentence = "取消订单"
result = predict_intent(sentence, model, tokenizer)
print(f"输入句子：{sentence}")
print(f"输出概率：{json.dumps(result, ensure_ascii=False, indent=2)}")