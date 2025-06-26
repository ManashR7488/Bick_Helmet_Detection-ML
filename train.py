### model train

from ultralytics import YOLO
model = YOLO('yolov11m.pt')
model.train(data='helmetDetectionDataset/data.yaml', epochs=50, imgsz=640, device=0)



# GPU test

# import torch
# print(torch.cuda.is_available())
