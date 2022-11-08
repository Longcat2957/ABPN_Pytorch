import torch
import torch_tensorrt
import cv2
import numpy as np
from libs.data_utils import npToTensor, openImage


INPUT_IMAGE_PATH = "./results/ms3_01.png"





if __name__ == "__main__":
    imgTensor = npToTensor(openImage(INPUT_IMAGE_PATH)).unsqueeze(0).float().cuda()
    
    qat_model = torch.jit.load("./weights/edgeSR_max_qat_40.jit.pt").eval()
    compile_spec = {
            "inputs" : [torch_tensorrt.Input([1, 3, 224, 320])],
            "enabled_precisions" : torch.int8,
            "truncate_long_and_double" : True
        }
    trt_model = torch_tensorrt.compile(qat_model, **compile_spec)
    sr_output = trt_model(imgTensor).squeeze(0).cpu().numpy().astype(np.uint8)
    srObj = np.transpose(sr_output, [1,2,0])
    srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
    while True:
        cv2.imshow("trt_demo.py", srObj)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cv2.destroyAllWindows()
    