from model import edgeSR
import torch
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

def q_wrapper(model:nn.Module, config:str="qnnpack"):
    model = model.eval()
    for i in range(len(model.feature_extraction)):
        model.feature_extraction[i].fuse_modules()
    qat_model = qatModel(model)
    qat_model.qconfig = torch.quantization.get_default_qconfig(backend=config)
    torch.quantization.prepare_qat(qat_model, inplace=True)
    return qat_model

if __name__ == "__main__":
    model = edgeSR()
    qat_model = q_wrapper(model)
    print(qat_model)