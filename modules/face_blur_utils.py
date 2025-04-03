# my_app/modules/face_blur_utils.py

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from stqdm import stqdm

from help_functions import create_video_writer, get_iou
from BlurFace import getBBoxFromVerifiedFace  # If this is the function you need from BlurFace.py


def detect_faces(img, detector, confidence=0.5):
    """
    Detect faces using a DNN-based approach (Caffe model).
    Returns list of bounding boxes.
    """
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    detector.setInput(blob)
    detections = detector.forward()
    bbox = []

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence:
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(np.int32)
            bbox.append(box)

    return bbox


def blur_image(model, blurred_image, prototxt, caffemodel, confidence=0.5):
    """
    Blurs faces in an image.
    If blurred_image == "All", blur all faces.
    If "Source Image", blur only the matched face(s).
    If "Everyone Else", blur all except matched face(s).
    """
    os.makedirs("images", exist_ok=True)
    detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    if blurred_image == "All":
        uploaded_verifying_image = st.file_uploader("Choose your input image...", type=["png", "jpg"], key=2)
        input_image_col, output_image_col = st.columns(2)

        if uploaded_verifying_image is not None:
            verifying_image = Image.open(uploaded_verifying_image)
            verifying_image = np.asarray(verifying_image, dtype=np.uint8)
            output_img = np.copy(verifying_image)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            bbox_list = detect_faces(output_img, detector, confidence)
            if bbox_list is not None:
                for box in bbox_list:
                    x1, y1, x2, y2 = box
                    img_blur_part = cv2.GaussianBlur(output_img[y1:y2, x1:x2], (49, 49), 0)
                    output_img[y1:y2, x1:x2] = img_blur_part

            with input_image_col:
                st.subheader("Your input Image")
                st.image(verifying_image, use_container_width=True)

            with output_image_col:
                st.subheader("Your output Image")
                output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                st.image(output_img_rgb, use_container_width=True)

    else:
        source_image_col, input_image_col = st.columns(2)
        with source_image_col:
            uploaded_source_images = st.file_uploader(
                "Choose your face you want to blur...",
                type=["png", "jpg"],
                key=1,
                accept_multiple_files=True
            )

        with input_image_col:
            uploaded_verifying_image = st.file_uploader(
                "Choose your input image...",
                type=["png", "jpg"],
                key=2
            )

        source_col, input_col, output_col = st.columns(3)

        if len(uploaded_source_images) > 0 and uploaded_verifying_image is not None:
            source_imgs = []
            for src in uploaded_source_images:
                src_img = Image.open(src)
                src_bgr = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
                source_imgs.append(src_bgr)

            verifying_image = Image.open(uploaded_verifying_image)
            verifying_image = np.asarray(verifying_image)
            output_img = np.copy(verifying_image)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            if blurred_image == "Source Image":
                for src_img in source_imgs:
                    bbox = getBBoxFromVerifiedFace(src_img, output_img, model)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        img_blur_part = cv2.GaussianBlur(output_img[y1:y2, x1:x2], (49, 49), 0)
                        output_img[y1:y2, x1:x2] = img_blur_part

            elif blurred_image == "Everyone Else":
                for src_img in source_imgs:
                    bbox_source = getBBoxFromVerifiedFace(src_img, output_img, model)
                    bbox_list = detect_faces(output_img, detector, confidence)

                    if bbox_source is not None and bbox_list is not None:
                        for box in bbox_list:
                            x1, y1, x2, y2 = box
                            iou = get_iou(box, bbox_source)
                            # If it's the same face, skip
                            if iou > 0.5:
                                continue

                            img_blur_part = cv2.GaussianBlur(output_img[y1:y2, x1:x2], (49, 49), 0)
                            output_img[y1:y2, x1:x2] = img_blur_part

            with source_col:
                st.subheader("Your source Image")
                # Convert back to RGB for display
                src_imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in source_imgs]
                st.image(src_imgs_rgb, use_container_width=True)

            with input_col:
                st.subheader("Your input Image")
                st.image(verifying_image, use_container_width=True)

            with output_col:
                st.subheader("Your output Image")
                output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                st.image(output_img_rgb, use_container_width=True)


def blur_video(model, blurred_image, prototxt, caffemodel, confidence=0.5):
    """
    Similar logic to blur_image, but for videos.
    """
    os.makedirs("videos", exist_ok=True)
    detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    if blurred_image == "All":
        uploaded_verifying_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        input_video_col, output_video_col = st.columns(2)

        if uploaded_verifying_video is not None:
            path = os.path.join("videos", uploaded_verifying_video.name)
            try:
                with open(path, "wb") as f:
                    f.write(uploaded_verifying_video.getbuffer())
                vid_file = True
            except PermissionError:
                vid_file = True

            with input_video_col:
                st.subheader("Your input Video")
                st.video(uploaded_verifying_video)

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

                        bbox_list = detect_faces(frame, detector, confidence)
                        if bbox_list is not None:
                            for box in bbox_list:
                                x1, y1, x2, y2 = box
                                img_blur_part = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 0)
                                frame[y1:y2, x1:x2] = img_blur_part

                        writer.write(frame)

                    cap.release()
                    writer.release()
                    video_file = open(output_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)

    else:
        source_image_col, input_video_col = st.columns(2)
        with source_image_col:
            uploaded_source_images = st.file_uploader(
                "Choose your face you want to blur...",
                type=["png", "jpg"],
                key=1,
                accept_multiple_files=True
            )
        with input_video_col:
            uploaded_verifying_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        source_col, input_col, output_col = st.columns(3)

        if len(uploaded_source_images) > 0 and uploaded_verifying_video is not None:
            # Convert source faces to BGR
            source_imgs = []
            for src in uploaded_source_images:
                src_img = Image.open(src)
                src_bgr = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
                source_imgs.append(src_bgr)

            path = os.path.join("videos", uploaded_verifying_video.name)
            try:
                with open(path, "wb") as f:
                    f.write(uploaded_verifying_video.getbuffer())
                vid_file = True
            except PermissionError:
                vid_file = True

            with source_col:
                st.subheader("Your source Image")
                src_imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in source_imgs]
                st.image(src_imgs_rgb, use_container_width=True)

            with input_col:
                st.subheader("Your input Video")
                st.video(uploaded_verifying_video)

            if vid_file:
                with output_col:
                    st.subheader("Your output Video")
                    cap = cv2.VideoCapture(path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    output_path = os.path.join("videos", "result.mp4")
                    writer = create_video_writer(cap, output_path)

                    for _ in stqdm(range(total_frames), colour="green"):
                        success, frame = cap.read()
                        if not success:
                            break

                        if blurred_image == "Source Image":
                            for src_img in source_imgs:
                                bbox = getBBoxFromVerifiedFace(src_img, frame, model)
                                if bbox:
                                    x1, y1, x2, y2 = bbox
                                    img_blur_part = cv2.GaussianBlur(frame[y1:y2, x1:x2], (49, 49), 0)
                                    frame[y1:y2, x1:x2] = img_blur_part

                        elif blurred_image == "Everyone Else":
                            for src_img in source_imgs:
                                bbox_source = getBBoxFromVerifiedFace(src_img, frame, model)
                                bbox_list = detect_faces(frame, detector, confidence)

                                if bbox_source is not None and bbox_list is not None:
                                    for box in bbox_list:
                                        x1, y1, x2, y2 = box
                                        iou_val = get_iou(box, bbox_source)
                                        if iou_val > 0.5:
                                            continue
                                        img_blur_part = cv2.GaussianBlur(frame[y1:y2, x1:x2], (49, 49), 0)
                                        frame[y1:y2, x1:x2] = img_blur_part

                        writer.write(frame)

                    cap.release()
                    writer.release()
                    video_file = open(output_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)
