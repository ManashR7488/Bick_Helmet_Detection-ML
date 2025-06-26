from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture("media/test1.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outvideo1.mp4", fourcc,
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    results = model.predict(source=frame, imgsz=640, conf=0.5)
    annotated = results[0].plot()
    out.write(annotated)

cap.release()
out.release()
