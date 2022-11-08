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

DEVICE = torch.device("cpu")
TEST_IMAGE = "./ms3_01.png"
FP32_WEIGHT = "./weights/1000_dts.pth"
QAT_WEIGHT = "./weights/qat_fbgemm_100.pth"

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
    fp32_model.load_state_dict(torch.load(FP32_WEIGHT, map_location="cpu"))
    
    fp32_model.eval()
    
    from libs.data_utils import *
    imgObj = openImage("in1.png")
    imgTensor = npToTensor(imgObj)
    
    cv2.imwrite("building_lr.jpg", cv2.cvtColor(imgObj, cv2.COLOR_RGB2BGR))

    with torch.no_grad():
        srTensor = fp32_model(imgTensor.unsqueeze(0).float()).squeeze(0)
    srObj = srTensor.numpy().astype(np.uint8)
    srObj = np.transpose(srObj, [1, 2, 0])
    srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
    # while True:
    #     cv2.imshow("test", srObj)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
        
    cv2.imwrite("building_fp32.jpg", srObj)
    # cv2.destroyAllWindows()
    
    
    bicubicTensor = tf.resize(imgTensor, (256 * 3, 256 * 3))
    bicubicObj = bicubicTensor.numpy().astype(np.uint8)
    bicubicObj = np.transpose(bicubicObj, [1, 2, 0])
    bicubicObj = cv2.cvtColor(bicubicObj, cv2.COLOR_RGB2BGR)
    cv2.imwrite("building_bicubic.jpg", bicubicObj)
    
    # fp32_qat_model = qat_wrapper(fp32_model)
    # fp32_qat_model.load_state_dict(torch.load(QAT_WEIGHT, map_location="cpu"))
    # fp32_qat_model.to('cuda')
    # fp32_qat_model.eval()
    # print(fp32_qat_model)
    
    
    # imgObj = openImage("ms3_01.png")
    # imgTensor = npToTensor(imgObj).float().unsqueeze(0)
    
    # with torch.no_grad():
    #     srTensor = fp32_qat_model(imgTensor.to('cuda')).squeeze(0)
    # srObj = srTensor.cpu().numpy().astype(np.uint8)
    # srObj = np.transpose(srObj, [1, 2, 0])
    # srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
    # while True:
    #     cv2.imshow("test", srObj)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
        
    # cv2.imwrite("output_qat_fbgemm_100_gpu.jpg", srObj)
    # # cv2.destroyAllWindows()
    
    # int8_model = qat_q_convert(fp32_qat_model)
    # int8_model.eval()
    # with torch.no_grad():
    #     srTensor = int8_model(imgTensor).squeeze(0)
    # srObj = srTensor.cpu().numpy().astype(np.uint8)
    # srObj = np.transpose(srObj, [1, 2, 0])
    # srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("output_Q_fbgemm_100_cpu.jpg", srObj)
    
