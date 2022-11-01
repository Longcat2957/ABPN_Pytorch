import torch
import torchvision
from libs.data_utils import *
from libs.model import *
from libs.qt import *
torch.backends.quantized.engine = 'fbgemm'
WEIGHT_PATH = "./weights/qat_fbgemm_final.pth"
q_model = get_q_model(WEIGHT_PATH)
#saveModel(q_model, "edgeSR_int8_qnnpack.pth")
net = torch.jit.script(q_model)
torch.jit.save(net, "edgeSR_int8_fbgemm_jit.pth")
# net2 = torch.jit.load("edgeSR_int8_qnnpack_jit.pth")
# a = torch.randn(size=(1, 3, 30, 30))
# b = net2(a)
# print(b.shape)