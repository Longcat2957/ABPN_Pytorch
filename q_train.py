import os
import argparse

from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from libs.data_utils import *
from libs.model import edgeSR
from libs.qt import qat_wrapper

DATA_ROOT = "../data/"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size", type=int, default=16
)
parser.add_argument(
    "--epochs", type=int, default=200
)
parser.add_argument(
    "--save_interval", type=int, default=30
)
parser.add_argument(
    "--weight", type=str, required=True, help="QAT NEEDS PRE-TRAINED FP32 Weight"
)
parser.add_argument(
    "--q_config", type=str, choices=["qnnpack", "fbgemm", "fbgemm_gpu"], default="qnnpack"
)

if __name__ == "__main__":
    opt = parser.parse_args()
    torch.backends.quantized.engine = opt.q_config
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"# DEVICE = {DEVICE}")
    traindataset = trainDataset(DATA_ROOT)
    valdataset = valDataset(DATA_ROOT)
    print(f"# TRAIN DATASET = {len(traindataset)}")
    print(f"# VALID DATASET = {len(valdataset)}")
    trainloader = DataLoader(
        traindataset, batch_size = opt.batch_size, shuffle=True, num_workers=os.cpu_count() - 2
    )
    valloader = DataLoader(
        valdataset, batch_size = opt.batch_size, shuffle=False, num_workers=os.cpu_count() - 2
    )
    model = edgeSR()
    model.to(DEVICE)
    try:
        model = loadModel(model, opt.weight)
        print(f"# Load Model .... ")
    except:
        raise ValueError(opt.weight)
    model = qat_wrapper(model, opt.q_config)    # model --> qat_model
    
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=[0.9, 0.999], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    for epoch in range(1, opt.epochs+1):
        # train
        model.train()
        trainmeter = tqdm(trainloader)
        for lr_tensor, hr_tensor in trainmeter:
            trainmeter.set_description(
                f"# TRAIN [{epoch}/{opt.epochs}]"
            )
            lr_tensor, hr_tensor = lr_tensor.to(DEVICE), hr_tensor.to(DEVICE)
            sr_pred = model(lr_tensor)
            optimizer.zero_grad()
            loss = criterion(sr_pred, hr_tensor)
            loss.backward()
            optimizer.step()
        
        # val
        model.eval()
        psnr_list, ssim_list = [], []
        for lr_tensor, hr_tensor in valloader:
            with torch.no_grad():
                lr_tensor = lr_tensor.to(DEVICE)
                sr_pred = model(lr_tensor).detach().cpu()
                psnr_value = float(peak_signal_noise_ratio(sr_pred, hr_tensor).item())
                ssim_value = float(structural_similarity_index_measure(sr_pred, hr_tensor).item())
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
            
        psnr_average, ssim_average = getAverage(psnr_list), getAverage(ssim_list)
        print(f"# VALIDATION RESULTS [{epoch}/{opt.epochs}] : PSNR = {psnr_average:.5f}, SSIM = {ssim_average:.5f}")
        scheduler.step()
        
        if epoch % opt.save_interval == 0:
            print(f"# SAVE WEIGHT")
            weight_name = f"qat_{opt.q_config}_{epoch}.pth"
            weight_name = os.path.join("./weights", weight_name)
            saveModel(model, weight_name)
    print(f"# SAVE FINAL RESULTS ...")
    weight_name = f"qat_{opt.q_config}_final.pth"
    weight_name = os.path.join("./weights", weight_name)
    saveModel(model, weight_name)