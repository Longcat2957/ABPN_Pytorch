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
TEST_IMAGE = "./ssd_sample.jpeg"
FP32_WEIGHT = "./weights/1000.pth"
QAT_WEIGHT = "./weights/qat_fbgemm_final.pth"

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
    # model = edgeSR().to(DEVICE)
    # model = loadModel(model, FP32_WEIGHT)
    # pilObj = Image.open(TEST_IMAGE)
    # preprocess = T.Compose([
    #     T.CenterCrop(size=(240, 426)),
    #     T.ToTensor()
    # ])
    # inputTensor = preprocess(pilObj)
    # inputTensor = inputTensor.unsqueeze(0).to(DEVICE)
    # # fp32 + gpu
    # inputTensor = inputTensor.to(torch.device("cuda"))
    # model.to(torch.device("cuda"))
    # model.eval()
    # with torch.no_grad():
    #     pred = model(inputTensor).detach().cpu()
    # fp32_gpu_time = calculateInferenceTime(model, inputTensor)
    # print(f">> fp32 + gpu ::{fp32_gpu_time * 1000:.3f}ms")
    
    # # fp16 + gpu
    # inputHalfTensor = inputTensor.half().to(torch.device("cuda"))
    # modelHalf = model.half().to(torch.device("cuda"))
    # fp16_gpu_time = calculateInferenceTime(modelHalf, inputHalfTensor)
    # print(f">> fp16 + gpu ::{fp16_gpu_time * 1000:.3f}ms")


    model = edgeSR().to(DEVICE)
    model = loadModel(model, FP32_WEIGHT)
    pilObj = Image.open(TEST_IMAGE)
    preprocess = T.Compose([
        T.CenterCrop(size=(240, 426)),
        T.ToTensor()
    ])
    inputTensor = preprocess(pilObj)
    inputTensor = inputTensor.unsqueeze(0).to(DEVICE)
    
    postprocess = T.Compose([
        T.ToPILImage()
    ])
    inputTensorToPilObj = postprocess(inputTensor.squeeze(0))
    inputTensorToPilObj.save("input.jpg", "JPEG")
    
    
    # fp32 + cpu
    fp32_cpu_inference_time = calculateInferenceTime(model, inputTensor)
    print(f">> fp32 + cpu ::{fp32_cpu_inference_time * 1000:.3f}ms")
    
    qat_model = qat_wrapper(model, config="fbgemm")
    qat_model.load_state_dict(torch.load(QAT_WEIGHT))
    # qat + cpu
    qat_inference_time = calculateInferenceTime(qat_model, inputTensor)
    print(f">> fake int8 + cpu :: {qat_inference_time * 1000:.3f}ms")
    
    # q + cpu
    q_model = qat_q_convert(qat_model, inplace=False)
    q_time = calculateInferenceTime(q_model, inputTensor)
    print(f">> int8 + cpu :: {q_time * 1000:.3f}ms")
    
    # Save results
    postprocess = T.Compose([
        T.ToPILImage()
    ])
    # post = postprocess(pred.squeeze(0))
    # post.save("output.jpg", "JPEG")
    