import torch
import torchvision

from model import social_stgcnn

dummy_v = torch.randn(1, 2, 8, 27, device="cuda")
dummy_a = torch.randn(8, 27, 27, device="cuda")

model = social_stgcnn(
    n_stgcnn=1, n_txpcnn=5, output_feat=5, seq_len=8, kernel_size=3, pred_seq_len=5,
).cuda()


input_names = ["vertices", "adjacency_kernel"]
output_names = ["vertices", "adjacency_kernel"]

torch.onnx.export(
    model,
    (dummy_v, dummy_a),
    "model.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    opset_version=12,
)
