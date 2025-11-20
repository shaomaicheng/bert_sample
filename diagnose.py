import torch
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer, BertForMaskedLM

# ----------------------------
# 1. 加载原始模型（不加任何多余参数）
# ----------------------------
print("🔍 正在加载 tokenizer 和模型...")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForMaskedLM.from_pretrained("bert-base-chinese")
model.eval()
print("✅ 模型加载完成")

# ----------------------------
# 2. 准备输入（实际长度，无 padding）
# ----------------------------
text = "北京是[MASK]国的首都"
inputs = tokenizer(text, return_tensors="pt")
print(f"\n📝 输入文本: {text}")
print(f"📥 input_ids shape: {inputs['input_ids'].shape}")
print(f"   input_ids: {inputs['input_ids']}")
print(f"   attention_mask: {inputs['attention_mask']}")

# 找 [MASK] 位置
mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
print(f"📍 [MASK] 位置: {mask_pos}")

# ----------------------------
# 3. PyTorch 推理
# ----------------------------
with torch.no_grad():
    pt_out = model(**inputs)
    pt_logits = pt_out.logits  # [1, seq, vocab]
    pt_probs = torch.softmax(pt_logits[0, mask_pos], dim=-1)
    pt_top5 = torch.topk(pt_probs, 5)

print("\n🟢 PyTorch 预测结果:")
for i in range(5):
    token_id = pt_top5.indices[i].item()
    prob = pt_top5.values[i].item()
    token = tokenizer.decode([token_id])
    print(f"  {i+1}. '{token}' (id={token_id}, prob={prob:.4f})")

# ----------------------------
# 4. ONNX 推理
# ----------------------------
try:
    ort_session = ort.InferenceSession("results/onnx/model.onnx")
except Exception as e:
    print(f"❌ ONNX 模型加载失败: {e}")
    exit(1)

ort_inputs = {
    "input_ids": inputs["input_ids"].numpy().astype(np.int64),
    "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
    "token_type_ids": inputs["token_type_ids"].numpy().astype(np.int64),
}

ort_out = ort_session.run(["logits"], ort_inputs)
ort_logits = ort_out[0]  # [1, seq, vocab]
from scipy.special import softmax
ort_probs = softmax(ort_logits[0, mask_pos])
top5_idx = np.argpartition(ort_probs, -5)[-5:]
top5_idx = top5_idx[np.argsort(-ort_probs[top5_idx])]

print("\n🔵 ONNX 预测结果:")
for i, idx in enumerate(top5_idx[:5]):
    token = tokenizer.decode([idx])
    print(f"  {i+1}. '{token}' (id={idx}, prob={ort_probs[idx]:.4f})")

# ----------------------------
# 5. 数值对比
# ----------------------------
pt_np = pt_logits.numpy()
diff = np.abs(pt_np - ort_logits).max()
print(f"\n📊 最大 logits 绝对误差: {diff:.6f}")

if diff < 1e-5:
    print("✅ 数值一致！ONNX 导出成功")
else:
    print("❌ 数值不一致！导出有问题")
    print(f"   PyTorch logits[{mask_pos}, :10] = {pt_np[0, mask_pos, :10]}")
    print(f"   ONNX    logits[{mask_pos}, :10] = {ort_logits[0, mask_pos, :10]}")