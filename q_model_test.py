import torch
import torchvision
from libs.data_utils import *
from libs.model import *
from libs.qt import *

WEIGHT_PATH = "./weights/qat_180.pth"
q_model = get_q_model(WEIGHT_PATH)
#saveModel(q_model, "edgeSR_int8_qnnpack.pth")
net = torch.jit.script(q_model)
# torch.jit.save(net, "edgeSR_int8_qnnpack_jit.pth")
net2 = torch.jit.load("edgeSR_int8_qnnpack_jit.pth")
print(net2)