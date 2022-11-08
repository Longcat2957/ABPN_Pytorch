import torch
import torchvision

qat_model = torch.jit.load("./weights/edgeSR_max_qat_200.jit.pt")
qat_model.cuda()
