import os
import cv2
import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms as T
from libs.model import edgeSR
from libs.qt import qat_wrapper, qat_q_convert

QAT_WEIGHT_PATH = "./weights/"

torch.backends.quantized.engine = 'fbgemm'
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 16)

preprocess = T.Compose([
    T.ToTensor()
])

net = edgeSR()
net = qat_wrapper(net, "fbgemm")
try:
    net.load_state_dict(torch.load(QAT_WEIGHT_PATH))
except:
    raise ValueError(QAT_WEIGHT_PATH)

net = qat_q_convert(net)
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")
        
        image = image[:, :, [2, 1, 0]]
        permuted = image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        out = net(input_batch)
        
        frame_count += 1
        now = time.time()
        # Count FPS per 1 seconds
        if now - last_logged > 1.0:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0