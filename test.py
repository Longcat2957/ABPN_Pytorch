import time
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.io import write_jpeg
import torchvision.transforms.functional as tf
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from libs.data_utils import *
from libs.model import *
from libs.qt import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TEST_IMAGE = "./results/ms3_01.png"
FP32_WEIGHT = "./weights/1000_dts.pth"

def calculateInferenceTime(model, input):
    model = model.eval()
    with torch.no_grad():
        for _ in range(50):
            o = model(input)
    
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            o = model(input)
            times.append(time.time() - start)
    
    return np.mean(times)
if __name__ == "__main__":
    fp32_model = edgeSR()
    fp32_model.load_state_dict(torch.load(FP32_WEIGHT, map_location="cpu"))
    fp32_model = fp32_model.cuda()
    fp32_model.eval()
    imgTensor = npToTensor(openImage(TEST_IMAGE)).unsqueeze(0).float().cuda()
    with torch.no_grad():
        srObj = fp32_model(imgTensor).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    srObj = np.transpose(srObj, [1,2,0])
    srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
    while True:
        cv2.imshow("test", srObj)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cv2.destroyAllWindows()
    a = calculateInferenceTime(fp32_model, imgTensor)
    print(f"# FP32 INFERENCE TIME = {a * 1000:.3f}ms")
    
    fp16_model = fp32_model.half()
    fp16_model = fp16_model.eval()
    imgTensor = npToTensor(openImage(TEST_IMAGE)).unsqueeze(0).half().cuda()
    
    b = calculateInferenceTime(fp16_model, imgTensor)
    print(f"# FP16 INFERENCE TIME = {b * 1000:.3f}ms")