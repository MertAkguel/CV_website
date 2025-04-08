# my_app/streamlit_pages/object_detection_page.py

import os
import streamlit as st
from ultralytics import YOLO

# Local imports from your modules:
from modules.detection_utils import prepare_classes
from modules.media_handlers import handle_image, handle_video, handle_webcam

def object_detection_page():
    """
    Renders the Object Detection page on Streamlit.
    """
    st.markdown("""
    <h1 style='text-align: center; color: white; font-size:400%;
    text-decoration-line: underline; text-decoration-color: red;'>
        Object Detection
    </h1>
    """, unsafe_allow_html=True)

    # If you need a model path, define or adjust here:
    model_path = r"C:\Users\Kleve\PycharmProjects\ComputerVision2\Resources"

    task = "detect"
    package = ""
    model = None
    classes_ids = []

    medium = st.sidebar.radio("Choose your medium", ["Image", "Video", "Webcam"])
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.01)
    model_select = st.sidebar.radio("Choose your Model", ["YOLO"])

    if model_select == "YOLO":
        package = "ultralytics"
        version = st.sidebar.selectbox(
            "Choose your version",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        )
        model = YOLO(os.path.join(model_path, version + ".pt"))
        classes_ids = prepare_classes(model)

    if medium == "Image":
        handle_image(model, classes_ids, task, package, confidence)
    elif medium == "Video":
        handle_video(model, classes_ids, task, package, confidence)
    elif medium == "Webcam":
        handle_webcam(model, classes_ids, task, package, confidence)
