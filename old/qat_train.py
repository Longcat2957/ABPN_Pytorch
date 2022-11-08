import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_tensorrt

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from libs.data_utils import *
from libs.model import *

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib
from tqdm import tqdm

DATA_ROOT = "../data/"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight", type=str, default="./weights/1000_dts.pth"
)
parser.add_argument(
    "--epochs", type=int, default=200
)
parser.add_argument(
    "--batch_size", type=int, default=16
)
parser.add_argument(
    "--preload", type=str, choices=["True", "False"], default="False"
)
parser.add_argument(
    "--save_interval", type=int, default=10
)

if __name__ == "__main__":
    # Set default QuantDescriptor
    quant_desc_input = QuantDescriptor(calib_method="max")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    # QAT init
    quant_modules.initialize()
    # print(pytorch_quantization.__version__)
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    opt = parser.parse_args()
    opt.preload = eval(opt.preload)
    
    # fp32+qat
    qat_model = edgeSR()
    # fp32 weight
    ckpt = torch.load(opt.weight)
    # load pretrained_weight
    qat_model.load_state_dict(ckpt)
    qat_model.to(DEVICE)
    print(qat_model)
    exit()
    
    # data_utils
    traindataset = trainDataset(DATA_ROOT, preload=opt.preload)
    valdataset = valDataset(DATA_ROOT, preload=opt.preload)
    print(f"# TRAIN DATASET = {len(traindataset)}")
    print(f"# VALID DATASET = {len(valdataset)}")
    trainloader = DataLoader(
        traindataset, batch_size = opt.batch_size, shuffle=True, num_workers=os.cpu_count() - 2,
        pin_memory=True
    )
    valloader = DataLoader(
        valdataset, batch_size = opt.batch_size, shuffle=False, num_workers=os.cpu_count() - 2,
        pin_memory=True
    )
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(qat_model.parameters(), lr=1e-4, betas=[0.9, 0.999], eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    for epoch in range(1, opt.epochs + 1):
        # train
        qat_model.train()
        trainmeter = tqdm(trainloader)
        for lr_tensor, hr_tensor in trainmeter:
            trainmeter.set_description(
                f"# <QAT> TRAIN [{epoch}/{opt.epochs}]"
            )
            lr_tensor, hr_tensor = lr_tensor.to(DEVICE), hr_tensor.to(DEVICE)
            sr_pred = qat_model(lr_tensor)
            optimizer.zero_grad()
            loss = criterion(sr_pred, hr_tensor)
            loss.backward()
            optimizer.step()
            
        # val
        qat_model.eval()
        psnr_list, ssim_list = [], []
        for lr_tensor, hr_tensor in valloader:
            with torch.no_grad():
                lr_tensor = lr_tensor.to(DEVICE)
                sr_pred = qat_model(lr_tensor).detach().cpu()
                psnr_value = float(peak_signal_noise_ratio(sr_pred, hr_tensor).item())
                ssim_value = float(structural_similarity_index_measure(sr_pred, hr_tensor).item())
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
            
        psnr_average, ssim_average = getAverage(psnr_list), getAverage(ssim_list)
        print(f"# VALIDATION RESULTS [{epoch}/{opt.epochs}] : PSNR = {psnr_average:.5f}, SSIM = {ssim_average:.5f}")
        scheduler.step()
        
        if epoch % opt.save_interval == 0:
            print(f"# SAVE WEIGHT")
            weight_name = f"qat_{epoch}.pth"
            weight_name = os.path.join("./weights", weight_name)
            saveModel(qat_model, weight_name)
    
    print(f"# SAVE FINAL RESULTS ...")
    weight_name = f"qat_final.pth"
    weight_name = os.path.join("./weights", weight_name)
    saveModel(qat_model, weight_name)
