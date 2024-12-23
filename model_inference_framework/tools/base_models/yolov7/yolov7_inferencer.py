import sys
sys.path.append("./tools/base_models/yolov7/")
from detect import detect
import torch

class Yolov7_Inferencer():
    def __init__(self, weight_path, source, size, save, save_txt, conf, stream):
        self.weight_path = weight_path
        self.size = size
        self.source = source
        self.save = save
        self.save_txt = save_txt
        self.conf = conf

    
    def yolov7_inference(self):
        with torch.no_grad():
            detect(self.weight_path, self.source, self.size, self.save, self.save_txt, self.conf)
        
    def yolov7_inference_with_results(self):
        self.just_results = True
        with torch.no_grad():
            bbox_arr, path_arr, class_arr = detect(self.weight_path, self.source, self.size, False, False, self.conf, self.just_results)

        return bbox_arr, path_arr, class_arr
