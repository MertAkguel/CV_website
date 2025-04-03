# my_app/pages/segmentation_page.py

import os
import streamlit as st
from ultralytics import YOLO, SAM

# Local imports
from modules.detection_utils import prepare_classes
from modules.media_handlers import handle_image, handle_video, handle_webcam

def segmentation_page():
    st.markdown("""
    <h1 style='text-align: center; color: white; font-size:400%;
    text-decoration-line: underline; text-decoration-color: red;'>
        Segmentation
    </h1>
    """, unsafe_allow_html=True)

    model_path = r"C:\Users\Kleve\PycharmProjects\ComputerVision2\Resources"

    task = "segment"
    package = ""
    model = None
    classes_ids = []

    medium = st.sidebar.radio("Choose your medium", ["Image", "Video", "Webcam"])
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.01)
    model_select = st.sidebar.radio("Choose your Model", ["YOLO", "SAM"])

    if model_select == "YOLO":
        package = "ultralytics"
        version = st.sidebar.selectbox(
            "Choose your version",
            ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"]
        )
        model = YOLO(os.path.join(model_path, version + ".pt"))
        classes_ids = prepare_classes(model)

    elif model_select == "SAM":
        package = "ultralytics"
        version = st.sidebar.selectbox(
            "Choose your version",
            ['sam_h', 'sam_l', 'sam_b', 'mobile_sam']
        )
        model = SAM(os.path.join(model_path, version + ".pt"))
        classes_ids = prepare_classes(model)

    if medium == "Image":
        handle_image(model, classes_ids, task, package, confidence)
    elif medium == "Video":
        handle_video(model, classes_ids, task, package, confidence)
    elif medium == "Webcam":
        handle_webcam(model, classes_ids, task, package, confidence)
