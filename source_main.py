import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained(
    "bert-base-chinese"
)
model.eval()
sentence = "北京是[MASK]国的首都"
# sentence = "北京是国的首都"

input_ids = tokenizer(sentence, return_tensors="pt")
# print(input_ids)
# print(tokenizer.mask_token_id)
mask_token_index=torch.where(input_ids["input_ids"] == tokenizer.mask_token_id)[1]
print(mask_token_index)

output = model(**input_ids)
# print(output)
logits = output.logits
print(logits)
print(logits.shape)
print(logits[0,mask_token_index,:].shape)
mask_logits = logits[0,mask_token_index,:]
mask_logits = torch.softmax(mask_logits, dim=1)
top_tokens = torch.topk(mask_logits, 5, dim=1).indices[0].tolist()
print(top_tokens)
for i,token in enumerate(top_tokens):
    word = tokenizer.decode(token)
    print(word)
    logit = mask_logits[0, token]
    print(logit.item() * 100)