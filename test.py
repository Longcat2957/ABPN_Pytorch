import time
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.io import write_jpeg
from libs.data_utils import *
from libs.model import *
from libs.qt import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")
TEST_IMAGE = "./ms3_01.png"
FP32_WEIGHT = "./weights/200.pth"
QAT_WEIGHT = "./weights/qat_180.pth"

def calculateInferenceTime(model, input):
    model = model.eval()
    with torch.no_grad():
        for i in range(10):
            o = model(input)
    start = time.time()
    with torch.no_grad():
        o = model(input)
        end = time.time()
    return end - start

if __name__ == "__main__":
    print(f"# DEVICE ? : {DEVICE}")
    fp32_model = edgeSR()
    fp32_model.load_state_dict(torch.load(FP32_WEIGHT))
    fp32_model.eval()
    
    inputPilObj = Image.open(TEST_IMAGE)
    preprocess = Compose([
        ToTensor()
    ])
    postprocess = Compose([
        ToPILImage()
    ])
    inputTensor = preprocess(inputPilObj).unsqueeze(0)
    with torch.no_grad():
        predTensor = fp32_model(inputTensor).squeeze(0)
    predPilObj = postprocess(predTensor)
    predPilObj.save("output_fp32_200.jpeg", "JPEG")
    
    fp32_model.clip = torch.nn.Identity()
    with torch.no_grad():
        predTensor = fp32_model(inputTensor).squeeze(0)
    predPilObj = postprocess(predTensor)
    predPilObj.save("output_fp32_200_noClip.jpeg", "JPEG")
    