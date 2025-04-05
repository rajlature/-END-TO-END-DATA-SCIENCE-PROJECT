import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

st.title("🧠 YOLOv8 Object Detection App")
st.write("Upload an image and see the detected objects using a pretrained YOLOv8 model!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to NumPy array
    image_np = np.array(image)

    # Run detection
    results = model(image_np)[0]

    # Draw detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image_np, caption='Detected Objects', use_column_width=True)
