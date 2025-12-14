

```markdown
# Real-time ASL Sign Detection with YOLOv8

This project implements a **real-time American Sign Language (ASL) word detector** using **YOLOv8** and a **Streamlit** web app. It covers the full pipeline from dataset preparation and training to an interactive webcam demo.

## Project Overview

- Fine-tunes **YOLOv8n** on an 80-class ASL dataset exported from **Roboflow**.
- Uses **YOLO object detection** to localize the signing region and classify the signed word.
- Provides a **Streamlit UI** for real-time detection from a webcam.
- Includes utilities for **dataset reduction** to enable faster CPU-only experimentation.

## Repository Structure

```
.
├── runs/                            # YOLO training runs (created after training)
├── src/
│   └── gesture_active_learning/
│       ├── data/
│       │   └── ASL.v1i.yolov8/      # Roboflow-exported YOLOv8 dataset (not tracked in git)
│       ├── models/
│       │   └── train_yolo.py        # YOLOv8 training script
│       ├── ui/
│       │   └── app.py               # Streamlit real-time detection app
│       └── utils/
│           └── reduce_dataset.py    # Script to create a smaller training subset
├── requirements.txt
├── pyproject.toml / Dockerfile      # (optional) reproducible environment
└── README.md
```

## Dataset

- Source: **ASL** word dataset hosted on Roboflow (80 restaurant-related ASL signs).  
- Classes (`nc = 80`):  
  `['additional', 'alcohol', 'allergy', 'bacon', 'bag', 'barbecue', 'bill', 'biscuit', 'bitter', 'bread', 'burger', 'bye', 'cake', 'cash', 'cheese', 'chicken', 'coke', 'cold', 'cost', 'coupon', 'credit card', 'cup', 'dessert', 'drink', 'drive', 'eat', 'eggs', 'enjoy', 'fork', 'french fries', 'fresh', 'hello', 'hot', 'icecream', 'ingredients', 'juicy', 'ketchup', 'lactose', 'lettuce', 'lid', 'manager', 'menu', 'milk', 'mustard', 'napkin', 'no', 'order', 'pepper', 'pickle', 'pizza', 'please', 'ready', 'receipt', 'refill', 'repeat', 'safe', 'salt', 'sandwich', 'sauce', 'small', 'soda', 'sorry', 'spicy', 'spoon', 'straw', 'sugar', 'sweet', 'thank-you', 'tissues', 'tomato', 'total', 'urgent', 'vegetables', 'wait', 'warm', 'water', 'what', 'would', 'yoghurt', 'your']`

**Note:** The dataset itself is not committed to this repository due to size and license. Download it from Roboflow and export in **YOLOv8 format** (images + TXT labels + `data.yaml`), then place it under:

```
src/gesture_active_learning/data/ASL.v1i.yolov8/
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
```

Update paths in `data.yaml` if needed.

## Environment Setup

```
# create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Key dependencies:

- `ultralytics` (YOLOv8)
- `torch`, `opencv-python`
- `streamlit`
- `numpy`, `pandas` (for utilities)

## Training the YOLOv8 Model

By default the training script uses the nano model (`yolov8n.pt`) and a reduced dataset for faster CPU experimentation.

```
python src/gesture_active_learning/models/train_yolo.py
```

This script:

- Loads `yolov8n.pt` via the Ultralytics API.
- Trains on `ASL.v1i.yolov8` (or a reduced subset).
- Writes results to `runs/detect/trainX/`.
- Saves the best weights to:

```
runs/detect/trainX/weights/best.pt
```

You can adjust key hyperparameters (epochs, image size, batch size, device) at the top of `train_yolo.py`.

### Optional: Reduce the Dataset

To create a smaller subset (e.g., 20% of images per split):

```
python src/gesture_active_learning/utils/reduce_dataset.py
```

This script copies a random fraction of images and labels into `ASL_small.yolov8/` and prints how many samples were kept per split.

## Real-Time Streamlit App

The app uses the trained YOLOv8 model to detect signs from your webcam.

1. Set the model path in `src/gesture_active_learning/ui/app.py` (or `config.py`):

```
MODEL_PATH = r"runs/detect/trainX/weights/best.pt"
CONF_THRESHOLD = 0.2
DEVICE = "cpu"   # or "cuda" if you have a GPU
CAM_INDEX = 0
```

2. Run Streamlit from the project root:

```
streamlit run src/gesture_active_learning/ui/app.py
```

3. In the browser:

- Check **“Start camera”**.
- Perform one of the trained ASL signs in front of the webcam.
- The app overlays YOLOv8 detections (bounding box + word label) and lists detected signs and confidences.



This project demonstrates:

- **Model training and tuning**  
  - Fine-tuning YOLOv8 on a custom multi-class sign language dataset.
  - Working with PyTorch / Ultralytics APIs and training on CPU/GPU.

- **Data engineering**  
  - Preparing Roboflow-exported data in YOLO format.
  - Implementing dataset reduction and experiment configuration.

- **Applied computer vision**  
  - Real-time inference on webcam streams with OpenCV.
  - Deployment of a human-facing UI using Streamlit.

- **Software engineering practices**  
  - Modular project layout (`models`, `ui`, `utils`).  
  - Reproducible environments (`requirements.txt`, optional Dockerfile).  
  - Clear documentation and paths for extension.

## Future Work

- Add temporal modeling for continuous sign sequences (e.g., 3D CNNs or Transformers).
- Support multi-signer scenarios and more diverse environments.
- Extend to additional ASL domains beyond restaurant vocabulary.

---

For questions or collaboration, feel free to open an issue or contact me.
```
