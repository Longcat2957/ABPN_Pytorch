import os
import cv2
import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms as T
from libs.model import edgeSR
from libs.qt import get_q_model, qat_wrapper, qat_q_convert

torch.backends.quantized.engine = "qnnpack"
QAT_WEIGHT_PATH = "./weights/qat_180.pth"
q_model = get_q_model(QAT_WEIGHT_PATH)
