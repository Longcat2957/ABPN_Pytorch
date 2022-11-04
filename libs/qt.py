from .model import edgeSR
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

def qat_wrapper(model:nn.Module, config:str="qnnpack"):
    model = model.eval()
    for i in range(len(model.feature_extraction)):
        model.feature_extraction[i].fuse_modules()
    qat_model = qatModel(model)
    qat_model.qconfig = torch.quantization.get_default_qconfig(backend=config)
    torch.quantization.prepare_qat(qat_model, inplace=True)
    return qat_model

def qat_q_convert(qat_model, inplace:bool=False, device:str="cpu"):
    return torch.quantization.convert(qat_model.to(torch.device(device)).eval(), inplace=inplace)

def get_q_model(qat_weight:str, config:str="qnnpack", map_location:str="cpu"):
    net = edgeSR()
    qat_model = qat_wrapper(net, config)
    qat_model.load_state_dict(torch.load(qat_weight, map_location=torch.device(map_location)))
    q_model = qat_q_convert(qat_model, inplace=False)
    return q_model

if __name__ == "__main__":
    torch.backends.quantized.engine = "qnnpack"
    model = edgeSR()
    qat_model = qat_wrapper(model)
    q_model = qat_q_convert(qat_model)
    net = torch.jit.script(q_model)