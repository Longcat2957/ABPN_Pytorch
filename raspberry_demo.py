import os
import cv2
import torch
import time
import numpy as np
from PIL import Image

torch.backends.quantized.engine = 'qnnpack'
