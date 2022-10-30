import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *

def isImage(filepath):
    try:
        _ = Image.open(filepath)
    except:
        return False
    return True

def openImage(filepath):
    try:
        imgObj = Image.open(filepath)
        return imgObj
    except:
        raise ValueError()

class trainDataset(Dataset):
    def __init__(self, root:str, lr_size:tuple=(360, 640), hr_size:tuple=(1080, 1920)) -> None:
        super().__init__()
        train_dir = os.path.join(root, 'train')
        assert os.path.exists(train_dir)
        self.file_list = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if isImage(os.path.join(train_dir, x))]
        
        self.hr_transform = Compose([
            RandomCrop(hr_size),
        ])
        self.lr_transform = Compose([
            Resize(size=lr_size)
        ])
        self.to_tensor = ToTensor()
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx:int) -> tuple:
        # return lr(tensor), hr(tensor)
        origImgObj = openImage(self.file_list[idx])
        
        hrPilObj = self.hr_transform(origImgObj)
        lrPilObj = self.lr_transform(hrPilObj)
        
        hrTensor = self.to_tensor(hrPilObj)
        lrTensor = self.to_tensor(lrPilObj)
        return lrTensor, hrTensor

class valDataset(Dataset):
    def __init__(self, root:str, lr_size:tuple=(360, 640), hr_size:tuple=(1080, 1920)) -> None:
        super().__init__()
        val_dir = os.path.join(root, 'val')
        assert os.path.exists(val_dir)
        self.file_list = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if isImage(os.path.join(val_dir, x))]
        
        self.hr_transform = Compose([
            CenterCrop(size=hr_size)
        ])
        self.lr_transform = Compose([
            Resize(size=lr_size)
        ])
        self.to_tensor = ToTensor()
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx) -> tuple:
        # return lr(tensor), hr(tensor)
        origImgObj = openImage(self.file_list[idx])
        
        hrPilObj = self.hr_transform(origImgObj)
        lrPilObj = self.lr_transform(hrPilObj)
        
        hrTensor = self.to_tensor(hrPilObj)
        lrTensor = self.to_tensor(lrPilObj)
        return lrTensor, hrTensor
    
class testDataset(Dataset):
    def __init__(self, root:str, lr_size:tuple=(360, 640), hr_size:tuple=(1080, 1920)) -> None:
        super().__init__()
        test_dir = os.path.join(root, 'test')
        assert os.path.exists(test_dir)
        self.file_list = [os.path.join(test_dir, x) for x in os.listdir(test_dir) if isImage(os.path.join(test_dir, x))]
        self.hr_transform = RandomCrop(size=hr_size)
        self.lr_transform = Resize(size=lr_size)
        self.sr_transform = Resize(size=hr_size)
        self.to_tensor = ToTensor()
        
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx) -> tuple:
        # return lr, sr, hr tensor
        origImgObj = openImage(self.file_list[idx])
        hrImgObj = self.hr_transform(origImgObj)
        lrImgObj = self.lr_transform(hrImgObj)
        srImgObj = self.sr_transform(lrImgObj)
        return self.to_tensor(lrImgObj), self.to_tensor(srImgObj), self.to_tensor(hrImgObj)
        
        
        
        