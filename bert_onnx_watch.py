
import onnxruntime as ort

sess = ort.InferenceSession("results/onnx/model.onnx")

# 输入
print("model.onnx输入")
for input in sess.get_inputs():
    print(f"{input.name}: shape={input.shape}, type={input.type}")

print("mode.onnx输出：")
for output in sess.get_outputs():
    print(f"{output.name}: shape={output.shape}, type={output.type}")