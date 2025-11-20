import onnxruntime as ort
from transformers import BertTokenizer, AutoTokenizer
import numpy as np
from scipy.special import softmax

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese',trust_remote_code=True)
session = ort.InferenceSession("results/onnx/model.onnx")

text = "北京是[MASK]国的首都"
inputs = tokenizer(text, return_tensors="np")  # ← NumPy

# 找 [MASK] 位置（用 NumPy）
mask_idx = np.where(inputs["input_ids"] == tokenizer.mask_token_id)[1][0]  # 取第一个

# 构建 ONNX 输入
ort_inputs = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
    "token_type_ids": inputs["token_type_ids"].astype(np.int64)
}

# 推理
logits = session.run(["logits"], ort_inputs)[0]  # [1, seq, vocab]
print(logits)

# 取 [MASK] 位置的 logits
mask_logits = logits[0, mask_idx, :]  # [vocab_size]

# 转概率 + top-k
probs = softmax(mask_logits)
top_k = 5
top_indices = np.argpartition(probs, -top_k)[-top_k:]
top_indices = top_indices[np.argsort(-probs[top_indices])]

for idx in top_indices:
    print(f"{tokenizer.decode([idx])}: {probs[idx]:.4f}")