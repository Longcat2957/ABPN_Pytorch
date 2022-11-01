import os
import cv2
import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms as T
from libs.model import edgeSR
from libs.qt import get_q_model, qat_wrapper, qat_q_convert


preprocess = T.Compose([
    T.CenterCrop(size=(240, 426)),
    T.ToTensor()
])
postprocess = T.Compose([
    T.ToPILImage()
])

# without quantization
# FP32_WEIGHT = "./weights/1000.pth"
# net = edgeSR()
# net.load_state_dict(torch.load(FP32_WEIGHT, map_location=torch.device("cpu")))
# net.eval()

# with torch.no_grad():
#     img = Image.open("input.jpeg")
#     started = time.time()
#     permuted = preprocess(img).unsqueeze(0)
#     pred = net(permuted)
#     time_elapsed = time.time() - started
#     output = postprocess(pred.squeeze(0))


# print(f"# BENCHMARK (SINGLE IMG) with fp32 [{time_elapsed * 1000:.3f}ms]")

# with quantization
def get_jit_model(weight_path:str):
    try:
        net = torch.jit.load(weight_path)
    except:
        raise RuntimeError(f"weight_path({weight_path}) is wrong")
    return net


torch.backends.quantized.engine = "qnnpack"
QAT_WEIGHT_PATH = "./weights/qat_qnnpack_1.pth"
q_model = get_q_model(QAT_WEIGHT_PATH)
# q_model = torch.jit.script(q_model)


with torch.no_grad():
    img = Image.open("input.jpeg")
    started = time.time()
    permuted = preprocess(img).unsqueeze(0)
    pred = q_model(permuted)
    time_elapsed = time.time() - started
    output = postprocess(pred.squeeze(0))

print(f"# BENCHMARK (SINGLE IMG) with Quantization [{time_elapsed * 1000:.3f}ms]")

# SHOW IMG
output.save("output_q.jpg", "JPEG")