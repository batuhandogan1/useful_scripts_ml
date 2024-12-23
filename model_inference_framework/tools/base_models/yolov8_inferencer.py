from ultralytics import YOLO

class Yolov8_Inferencer():
    def __init__(self, weight_path, source, size, save, save_txt, conf, stream):
        self.weight_path = weight_path
        self.size = size
        self.source = source
        self.save = save
        self.save_txt = save_txt
        self.conf = conf
        self.stream = stream

    
    def yolov8_inference(self):
        model = YOLO(self.weight_path, task='detect')
        results = model.predict(source=self.source, imgsz=self.size, save=self.save, save_txt=self.save_txt, conf=self.conf, stream=self.stream)
        for result in results:
            print(result)

        
    def yolov8_inference_with_results(self):
        bbox_arr = []
        path_arr = []
        class_arr = []

        model = YOLO(self.weight_path, task='detect')
        results = model(source=self.source, imgsz=self.size, conf=self.conf, stream=self.stream)
        
        for result in results:

            for i, box in enumerate(result.boxes.xywhn):
                bbox_arr.append(box.tolist())
                path_arr.append(result.path)
                class_arr.append(result.names[int(result.boxes.cls[i])])

        return bbox_arr, path_arr, class_arr
