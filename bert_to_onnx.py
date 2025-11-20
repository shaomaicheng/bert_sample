import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertForMaskedLM


def main():



    #加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    #加载权重
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    model.eval()

    # 输入
    text='北京是[MASK]国的首都'
    inputs = tokenizer(text, return_tensors='pt')
    print(list(inputs.keys()))
    outputs = model(**inputs)
    print(list(outputs.keys()))
    input_names = ['input_ids','attention_mask', 'token_type_ids']
    output_names = ['logits']
    #导出onnx
    torch.onnx.export(
        model,
        args=(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        f='results/onnx/model.onnx',
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=14,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False
    )
    print('onnx model saved!')



if __name__ == "__main__":
    main()