from ultralytics import YOLO

model = YOLO(r"runs/detect/train7/weights/best.pt")
results = model(r"src/gesture_active_learning/data/ASL_small.yolov8/train/images/-ePyJBmOJ9s_mp4-9_jpg.rf.955627e9e0ed163b9927d74ab0c324d3.jpg", conf=0.1)
results[0].show()
print(results[0].boxes.cls, results[0].boxes.conf)
