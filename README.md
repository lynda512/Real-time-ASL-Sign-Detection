
# Real-time ASL Sign Detection with YOLOv8

This project implements a **real-time American Sign Language (ASL) word detector** using **YOLOv8** and a **Streamlit** web app. It covers the full pipeline from dataset preparation and training to an interactive webcam demo.

## Project Overview

- Fine-tunes **YOLOv8n** on an 80-class ASL dataset exported from **Roboflow**.
- Uses **YOLO object detection** to localize the signing region and classify the signed word.
- Provides a **Streamlit UI** for real-time detection from a webcam.
- Includes utilities for **dataset reduction** to enable faster CPU-only experimentation.
  ![]procjte.gif

## Repository Structure


.
├── runs/                         # YOLO training runs (created after training)
├── src/
│   └── gesture_active_learning/
│       ├── data/
│       │   └── ASL.v1i.yolov8/   # Roboflow-exported YOLOv8 dataset (not tracked in git)
│       ├── models/
│       │   └── train_yolo.py     # YOLOv8 training script
│       ├── ui/
│       │   └── app.py            # Streamlit real-time detection app
│       └── utils/
│           └── reduce_dataset.py # Script to create a smaller training subset
├── requirements.txt
├── pyproject.toml / Dockerfile   # (optional) reproducible environment
└── README.md


## Dataset

- Source: **ASL** word dataset hosted on Roboflow (80 restaurant-related ASL signs).
- Classes (`nc = 80`):


['additional', 'alcohol', 'allergy', 'bacon', 'bag', 'barbecue', 'bill', 'biscuit',
 'bitter', 'bread', 'burger', 'bye', 'cake', 'cash', 'cheese', 'chicken', 'coke',
 'cold', 'cost', 'coupon', 'credit card', 'cup', 'dessert', 'drink', 'drive', 'eat',
 'eggs', 'enjoy', 'fork', 'french fries', 'fresh', 'hello', 'hot', 'icecream',
 'ingredients', 'juicy', 'ketchup', 'lactose', 'lettuce', 'lid', 'manager', 'menu',
 'milk', 'mustard', 'napkin', 'no', 'order', 'pepper', 'pickle', 'pizza', 'please',
 'ready', 'receipt', 'refill', 'repeat', 'safe', 'salt', 'sandwich', 'sauce',
 'small', 'soda', 'sorry', 'spicy', 'spoon', 'straw', 'sugar', 'sweet', 'thank-you',
 'tissues', 'tomato', 'total', 'urgent', 'vegetables', 'wait', 'warm', 'water',
 'what', 'would', 'yoghurt', 'your']


**Note:** The dataset itself is not committed to this repository due to size and license.  
Download it from Roboflow and export in **YOLOv8 format** (images + TXT labels + `data.yaml`), then place it under:


src/gesture_active_learning/data/ASL.v1i.yolov8/
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels


Update paths in `data.yaml` if needed.

## Environment Setup


# create and activate virtual environment (optional)
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt


Key dependencies:

- `ultralytics` (YOLOv8)
- `torch`, `opencv-python`
- `streamlit`
- `numpy`, `pandas` (for utilities)

## Training the YOLOv8 Model

By default the training script uses the nano model (`yolov8n.pt`) and a reduced dataset for faster CPU experimentation.


python src/gesture_active_learning/models/train_yolo.py


This script:

- Loads `yolov8n.pt` via the Ultralytics API.
- Trains on the ASL dataset (or the reduced subset) using the settings defined in the script.
- Writes results to `runs/detect/trainX/`.
- Saves the best weights to:


runs/detect/trainX/weights/best.pt


You can tune hyperparameters such as `epochs`, `imgsz`, `batch`, and `device` at the top of `train_yolo.py`.

### Optional: Reduce the Dataset

To speed up CPU-only experiments, you can create a smaller version of the dataset:


python src/gesture_active_learning/utils/reduce_dataset.py


This copies a fraction of images and labels into `ASL_small.yolov8/` and prints how many samples were kept per split.  
Point your training `data.yaml` to this smaller dataset when debugging.

## Model Performance

The trained model is evaluated on the validation set using YOLOv8’s built-in metrics.

- **mAP@0.5:** ≈ **0.98** across all 80 classes.
 <img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/fc5cbae7-d9a6-411f-b596-d324d7217d26" />

- **Best F1 score:** ≈ **0.97** at an optimal confidence threshold around **0.6**.
 <img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/d853f3b6-a0b2-4ee5-835f-353dabad7b7c" />

- **Precision–Recall curves:** The aggregated curve stays close to the top edge, indicating high precision and recall over a wide range of thresholds.
  <img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/1c7961f1-97c8-459b-a25a-ac0fbc1b676f" />

- **Confusion matrices (normalized and raw):** Almost purely diagonal with only a few off-diagonal cells, showing that different signs are rarely confused on the validation set.
  <img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c806fc7b-a9c7-4669-a49e-abb5f4d0d1f7" />

- **Training curves:** Box, classification, and DFL losses decrease smoothly; precision, recall, mAP@0.5 and mAP@0.5–0.95 increase and plateau without clear overfitting.

In practice, real-time performance depends on how similar the webcam setup is to the training images (camera distance, lighting, framing) and on the chosen confidence threshold in the app. Lower thresholds (e.g. `0.2–0.3`) make the detector more sensitive; higher thresholds make it stricter but may miss some signs.

## Real-Time Streamlit App

The Streamlit app provides a simple interface to run the trained model on a live webcam feed.

1. Set the model path and basic config in `src/gesture_active_learning/ui/app.py` (or `config.py`):


MODEL_PATH = r"runs/detect/trainX/weights/best.pt"
CONF_THRESHOLD = 0.25           # default confidence for UI slider
DEVICE = "cpu"                  # or "cuda" if you have a GPU
CAM_INDEX = 0                   # default webcam


2. Launch the app from the project root:


streamlit run src/gesture_active_learning/ui/app.py


3. In the browser:

- Check **“Start camera”**.
- Perform one of the 80 trained ASL signs in front of the webcam.
- The app shows the webcam stream with YOLOv8 bounding boxes and labels, and lists detected signs with their confidences.

If detections seem sparse, reduce the confidence slider in the sidebar and make sure your pose and distance from the camera roughly match the dataset examples.

## Highlights

- End-to-end computer vision pipeline: data preparation, model training, evaluation, and interactive deployment.
- Practical usage of PyTorch/Ultralytics YOLO, Streamlit, and OpenCV in a modular Python codebase.
- Designed to be a small but complete example of applying modern object detection to sign language recognition.
