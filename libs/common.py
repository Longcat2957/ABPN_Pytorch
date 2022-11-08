import torch
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def getAverage(List:list):
    length = len(List)
    summation = 0.
    for item in List:
        summation += item
    return summation / length

def train_one_epoch(
    model:nn.Module,
    trainloader:DataLoader,
    valloader:DataLoader,
    criterion:nn.modules.loss,
    optimizer:optim.Optimizer,
    lr_scheduler:optim.lr_scheduler,
):
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Train Phase
    model.to(DEVICE)
    model.train()
    
    #wrapping with tqdm
    trainloader = tqdm(trainloader)
    for lr_tensor, hr_tensor in trainloader:
        lr_tensor, hr_tensor = lr_tensor.to(DEVICE), hr_tensor.to(DEVICE)
        sr_tensor = model(lr_tensor)
        optimizer.zero_grad()
        loss = criterion(sr_tensor, hr_tensor)
        loss.backward()
        optimizer.step()
    
    # VAL PHASE
    model.eval()
    psnr_list, ssim_list = [], []
    for lr_tensor, hr_tensor in valloader:
        with torch.no_grad():
            lr_tensor = lr_tensor.to(DEVICE)
            sr_pred = model(lr_tensor).detach().cpu()
            psnr_value = float(
                peak_signal_noise_ratio(
                    sr_pred, hr_tensor
                )
            )
            ssim_value = float(
                structural_similarity_index_measure(
                    sr_pred, hr_tensor
                )
            )
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
    psnr_average, ssim_average = getAverage(psnr_list), getAverage(ssim_list)
    print(f"# VALIDATION RESULTS : PSNR = {psnr_average:.5f}, SSIM = {ssim_average:.5f}")
    lr_scheduler.step()