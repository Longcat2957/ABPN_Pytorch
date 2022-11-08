import torch
from libs.data_utils import *
from libs.model import *

model = edgeSR()
model.load_state_dict(torch.load("./weights/1000_dts.pth"))
model.eval()
dummy = torch.randn((1, 3, 224, 320))
torch.onnx.export(model, dummy, "test.onnx")