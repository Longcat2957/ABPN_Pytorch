import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import *

def saveModel(model:nn.Module, name:str):
    assert not os.path.exists(name), FileExistsError(f"{name} is exists")
    torch.save(model.state_dict(), name)

def loadModel(model:nn.Module, weight:str, device:str='cpu'):
    assert os.path.exists(weight), FileNotFoundError(f"{weight} not found")
    model.load_state_dict(torch.load(weight, map_location=torch.device(device)))
    return model

def isImage(filepath):
    try:
        _ = Image.open(filepath)
    except:
        return False
    return True


# def openImage(filepath):
#     try:
#         imgObj = Image.open(filepath)
#         return imgObj
#     except:
#         raise ValueError()

def openImage(filepath):
    try:
        imgObj = cv2.imread(filepath, cv2.IMREAD_COLOR)
        imgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2RGB)
        return imgObj
    except:
        raise ValueError()

def npToTensor(x:np.ndarray):
    x = np.transpose(x, [2, 0, 1])
    tensor = torch.from_numpy(x)
    return tensor

def getAverage(List:list):
    length = len(List)
    summation = 0.
    for item in List:
        summation += item
    return summation / length

class trainDataset(Dataset):
    def __init__(self, root:str, lr_size:tuple=(64, 64), hr_size:tuple=(64*3, 64*3), \
        preload:bool=True) -> None:
        super().__init__()
        train_dir = os.path.join(root, 'train')
        assert os.path.exists(train_dir)
        self.file_list = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if isImage(os.path.join(train_dir, x))]
        self.preload = preload
        if preload:
            self.preloaded = [npToTensor(openImage(x)) for x in self.file_list]
        else:
            self.preloaded = None
        self.hr_transform = Compose([
            RandomCrop(hr_size),
            RandomHorizontalFlip(),
            RandomRotation(180)
        ])
        self.lr_transform = Compose([
            Resize(size=lr_size)
        ])
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx:int) -> tuple:
        if self.preload:
            origImgTensor = self.preloaded[idx]
        else:
            origImgObj = openImage(self.file_list[idx]) # origImgObj is np.ndarray
            origImgTensor = npToTensor(origImgObj)      #
        
        hrTensor = self.hr_transform(origImgTensor)
        lrTensor = self.lr_transform(hrTensor)
        
        return lrTensor.float(), hrTensor.float()

class valDataset(Dataset):
    def __init__(self, root:str, lr_size:tuple=(64, 64), hr_size:tuple=(64*3, 64*3), preload:bool=True) -> None:
        super().__init__()
        val_dir = os.path.join(root, 'val')
        assert os.path.exists(val_dir)
        self.file_list = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if isImage(os.path.join(val_dir, x))]
        self.preload = preload
        if preload:
            self.preloaded = [npToTensor(openImage(x)) for x in self.file_list]
        else:
            self.preloaded = None
        self.hr_transform = Compose([
            CenterCrop(size=hr_size)
        ])
        self.lr_transform = Compose([
            Resize(size=lr_size)
        ])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx) -> tuple:
        if self.preload:
            origImgTensor = self.preloaded[idx]
        else:
            origImgObj = openImage(self.file_list[idx]) # origImgObj is np.ndarray
            origImgTensor = npToTensor(origImgObj)      
        
        hrTensor = self.hr_transform(origImgTensor)
        lrTensor = self.lr_transform(hrTensor)

        return lrTensor.float(), hrTensor.float()
    
class testDataset(Dataset):
    def __init__(self, root:str, lr_size:tuple=(360, 640), hr_size:tuple=(1080, 1920)) -> None:
        super().__init__()
        test_dir = os.path.join(root, 'test')
        assert os.path.exists(test_dir)
        self.file_list = [os.path.join(test_dir, x) for x in os.listdir(test_dir) if isImage(os.path.join(test_dir, x))]
        self.valid_file_list = []
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
    
    
if __name__ == "__main__":
    imgpath = "../ms3_01.png"
    imgObj = openImage(imgpath)
    # while True:
    #     cv2.imshow("read?", cv2.cvtColor(imgObj, cv2.COLOR_RGB2BGR))
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
    # cv2.destroyAllWindows()
    imgTensor = npToTensor(imgObj)
