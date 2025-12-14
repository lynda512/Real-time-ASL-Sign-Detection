# app.py

import streamlit as st
import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = r"runs/detect/train7/weights/best.pt"
CONF_THRESHOLD = 0.1
DEVICE = "cpu"
CAM_INDEX = 0

st.set_page_config(page_title="YOLOv8 Sign Detection", layout="wide")
st.title("Real-time Sign Language Detection (YOLOv8)")

@st.cache_resource
def load_model(path):
    model = YOLO(path)
    model.to(DEVICE if torch.cuda.is_available() else "cpu")
    return model

model = load_model(MODEL_PATH)
names = model.names

run = st.checkbox("Start camera", value=False, key="run_cam")
conf_slider = st.slider(
    "Confidence threshold", 0.1, 0.9, CONF_THRESHOLD, 0.05
)

frame_ph = st.empty()
label_ph = st.empty()

if run:
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("No frame from camera.")
                break

            results = model.predict(
                frame,
                conf=conf_slider,
                verbose=False
            )
            res = results[0]
            annotated = res.plot()

            detected = []
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names[cls_id]
                detected.append(f"{label} ({conf:.2f})")

            frame_ph.image(annotated[:, :, ::-1], channels="RGB")

            if detected:
                label_ph.markdown("### Detected signs: " + ", ".join(detected))
            else:
                label_ph.markdown("### No sign detected.")

            # stop when user unchecks box
            if not st.session_state.get("run_cam", False):
                break

        cap.release()
else:
    st.info("Check 'Start camera' to begin detection.")
