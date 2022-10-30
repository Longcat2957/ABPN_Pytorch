from model import edgeSR
import torch.nn as nn
import torch.quantization as tq

class qatModel(nn.Module):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)    
        return x

def q_wrapper(model:nn.Module):
    model = model.eval()
    for i in range(len(model.feature_extraction)):
        model.feature_extraction[i].fuse_modules()
    return qatModel(model)


if __name__ == "__main__":
    model = edgeSR()
    qm = q_wrapper(model)
    print(qm)