# my_app/modules/media_handlers.py

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from stqdm import stqdm

from modules.detection_utils import predict_and_detect, predict_and_segment
from help_functions import create_video_writer


def handle_image(chosen_model, classes, task, package, conf=0.5):
    os.makedirs("images", exist_ok=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])

    input_image_col, output_image_col = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        new_img = np.array(image.convert('RGB'))

        with input_image_col:
            st.subheader("Your input Image")
            st.image(image, use_container_width=True)

        with output_image_col:
            st.subheader("Your output image")

            if package == "ultralytics":
                if task == "detect":
                    result_img, _ = predict_and_detect(chosen_model, new_img, classes, conf)
                elif task == "segment":
                    result_img, _ = predict_and_segment(chosen_model, new_img, classes, conf)

                st.image(result_img, use_container_width=True)
                result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite("images/result_img.png", result_img_bgr)

            elif package == "super_gradients":
                chosen_model.predict(new_img, conf=conf).save("images")
                # e.g., load the first file from "images" as output
                output_filename = os.listdir("images")[0]
                result_img = cv2.imread(os.path.join("images", output_filename))
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img, use_container_width=True)

        # Download button
        with open("images/result_img.png", "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name='result_img.png',
                mime='image/png',
            )


def handle_video(chosen_model, classes, task, package, conf=0.5):
    os.makedirs("videos", exist_ok=True)
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

    input_video_col, output_video_col = st.columns(2)

    if uploaded_file is not None:

        path = os.path.join("videos", uploaded_file.name)
        try:
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            vid_file = True
        except PermissionError:
            # If there's some OS-level lock, handle gracefully
            vid_file = True

        with input_video_col:
            st.subheader("Your input Video")
            st.video(uploaded_file)

        if vid_file:
            with output_video_col:
                st.subheader("Your output Video")

                cap = cv2.VideoCapture(path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                output_path = os.path.join("videos", "result.mp4")

                writer = create_video_writer(cap, output_path)

                for _ in stqdm(range(total_frames), colour="green"):
                    success, frame = cap.read()
                    if not success:
                        break

                    if package == "ultralytics":
                        if task == "detect":
                            result_img, _ = predict_and_detect(chosen_model, frame, classes, conf)
                        elif task == "segment":
                            result_img, _ = predict_and_segment(chosen_model, frame, classes, conf)

                    elif package == "super_gradients":
                        chosen_model.predict(frame, conf=conf).save("images")
                        file_name = os.listdir("images")[0]
                        result_img = cv2.imread(os.path.join("images", file_name))

                    writer.write(result_img)

                cap.release()
                writer.release()

                video_file = open(output_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)

        with open("videos/result.mp4", "rb") as file:
            st.download_button(
                label="Download Video",
                data=file,
                file_name='result.mp4',
            )


def handle_webcam(chosen_model, classes, task, package, conf=0.5):
    os.makedirs("webcam", exist_ok=True)
    col1, col2 = st.columns(2)

    start = col1.button("Start")
    stop = col2.button("Stop")

    camera = cv2.VideoCapture(0)
    input_webcam_col, output_webcam_col = st.columns(2)

    with input_webcam_col:
        st.subheader("Input of Webcam")
        frame_window_input = st.image([])

    with output_webcam_col:
        st.subheader("Output of Webcam")
        frame_window_output = st.image([])

    # Simple loop approach. In real usage, might use while True & break if stop
    while start:
        success, frame = camera.read()
        if not success:
            st.warning("Could not read webcam frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window_input.image(frame_rgb)

        result_img = ""
        # Apply detection/segmentation if desired
        if package == "ultralytics":
            if task == "detect":
                result_img, _ = predict_and_detect(chosen_model, frame_rgb, classes, conf)
            elif task == "segment":
                result_img, _ = predict_and_segment(chosen_model, frame_rgb, classes, conf)

            frame_window_output.image(result_img)

        elif package == "super_gradients":
            chosen_model.predict(frame_rgb, conf=conf).save("webcam")
            file_name = os.listdir("webcam")[0]
            result_img = cv2.imread(os.path.join("webcam", file_name))
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            frame_window_output.image(result_img_rgb)

        if stop:
            break

    camera.release()
