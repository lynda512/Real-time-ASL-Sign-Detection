# src/gesture_active_learning/models/train_yolo.py

from ultralytics import YOLO

# path to your dataset YAML (use forward slashes or raw string)
DATA_YAML = r"src/gesture_active_learning/data/ASL_small.yolov8/data.yaml"

def main():
    # load a small YOLOv8 model
    
    model = YOLO("yolov8n.pt")
    model.train(
    data=DATA_YAML,
    epochs=3,          # very small just to verify pipeline
    imgsz=320,         # much smaller images
    batch=4,           # small batch for low RAM/CPU
    workers=0,         # Windows + CPU
    device="cpu"       # explicit
)




    # best.pt is already saved under runs/detect/train/weights
    # if you want a copy in this folder:
    model.export(format="pt")  # optional

if __name__ == "__main__":
    main()
