import streamlit as st
import random
import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from PIL import Image
from stqdm import stqdm
from torch import cuda
from help_functions import create_video_writer, get_iou
from super_gradients.training import models
from BlurFace import *


def prepare_classes(model):
    yolo_classes = list(model.names.values())
    classes = st.sidebar.multiselect(
        'Select your classes',
        ['All'] + sorted(yolo_classes),
        ['All'])

    if 'All' not in classes:
        classes_ids = [yolo_classes.index(clas) for clas in classes]
    else:
        classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    return classes_ids


def predict(chosen_model, img, classes, conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes, conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results


def predict_and_segment(chosen_model, img, classes, conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes]

    try:
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                # cv2.polylines(img, points, True, (255, 0, 0), 1)
                color_number = classes.index(int(box.cls[0]))
                cv2.fillPoly(img, points, colors[color_number])
    except Exception as e:
        print(e)

    return img, results


def handle_image(chosen_model, classes, task, package, conf=0.5):
    os.makedirs("images", exist_ok=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])

    input_image, output_image = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        new_img = np.array(image.convert('RGB'))

        with input_image:
            st.subheader("Your input Image")
            st.image(image, use_column_width=True)

        with output_image:
            st.subheader("Your output image")

            if package == "ultralytics":
                if task == "detect":

                    result_img, _ = predict_and_detect(chosen_model, new_img, classes, conf)
                elif task == "segment":
                    result_img, _ = predict_and_segment(chosen_model, new_img, classes, conf)

                st.image(result_img, use_column_width=True)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite("images/result_img.png", result_img)

            elif package == "super_gradients":

                chosen_model.predict(new_img, conf=conf).save("images")
                result_img = cv2.imread(os.path.join("images", os.listdir("images")[0]))
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img, use_column_width=True)

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

    input_video, output_video = st.columns(2)

    if uploaded_file is not None:

        path = os.path.join("videos", uploaded_file.name)
        # vid_file = False
        try:
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                vid_file = True
        except PermissionError:
            vid_file = True
            pass

        with input_video:
            st.subheader("Your input Video")
            st.video(uploaded_file)

        if vid_file:
            with output_video:
                st.subheader("Your output Video")

                cap = cv2.VideoCapture(path)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_path = os.path.join("videos", "result.mp4")

                writer = create_video_writer(cap, output_path)

                for _ in stqdm(range(total_frames), colour="green"):

                    success, image = cap.read()

                    if not success:
                        break

                    if package == "ultralytics":
                        if task == "detect":
                            result_img, _ = predict_and_detect(chosen_model, image, classes, conf)
                        elif task == "segment":
                            result_img, _ = predict_and_segment(chosen_model, image, classes, conf)

                    elif package == "super_gradients":

                        chosen_model.predict(image, conf=conf).save("images")
                        result_img = cv2.imread(os.path.join("images", os.listdir("images")[0]))

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
                # mime='image/png',
            )


def handle_webcam(chosen_model, classes, task, package, conf=0.5):
    os.makedirs("webcam", exist_ok=True)
    col1, col2 = st.columns(2)

    with col1:
        start = st.button("Start")
    with col2:
        stop = st.button("Stop")

    camera = cv2.VideoCapture(0)

    input_webcam, output_webcam = st.columns(2)

    with input_webcam:
        st.subheader("Input of Webcam")
        FRAME_WINDOW_INPUT = st.image([])

    with output_webcam:
        st.subheader("Output of Webcam")
        FRAME_WINDOW_OUTPUT = st.image([])

    while start:
        _, frame = camera.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with input_webcam:

            FRAME_WINDOW_INPUT.image(frame)

        with output_webcam:

            if package == "ultralytics":
                if task == "detect":
                    result_img, _ = predict_and_detect(chosen_model, frame, classes, conf)
                elif task == "segment":
                    result_img, _ = predict_and_segment(chosen_model, frame, classes, conf)
                FRAME_WINDOW_OUTPUT.image(result_img)

            elif package == "super_gradients":

                chosen_model.predict(frame, conf=conf).save("webcam")
                result_img = cv2.imread(os.path.join("webcam", os.listdir("webcam")[0]))
                FRAME_WINDOW_OUTPUT.image(result_img)

        if stop:
            break


def detect_faces(img, detector, confidence=0.5):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()
    bbox = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        conf = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if conf > confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(np.int)
            bbox.append(box)

    return bbox


def blur_image(model, blurred_image, prototxt, caffemodel, confidence=0.5):
    global x_face_recog, y_face_recog, w_face_recog, h_face_recog
    os.makedirs("images", exist_ok=True)
    detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    if blurred_image == "All":

        uploaded_verifiing_image = st.file_uploader("Choose your input image...", type=["png", "jpg"], key=2)
        input_image, output_image = st.columns(2)

        if uploaded_verifiing_image is not None:
            # verifiing_image = Image.open(uploaded_verifiing_image)
            # verifiing_image = np.asarray(verifiing_image.convert('RGB'))
            verifiing_image = Image.open(uploaded_verifiing_image)
            verifiing_image = np.asarray(verifiing_image, dtype=np.uint8)
            # st.write(f"{verifiing_image.shape = }")
            output_img = np.copy(verifiing_image)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            bbox = detect_faces(output_img, detector, confidence)
            if bbox is not None:

                for box in bbox:
                    imgBlurPart = cv2.GaussianBlur(output_img[box[1]:box[3],
                                                   box[0]:box[2]],
                                                   (49, 49), 0)
                    output_img[box[1]:box[3],
                    box[0]:box[2]] = imgBlurPart

            with input_image:
                st.subheader("Your input Image")
                st.image(verifiing_image, use_column_width=True)

            with output_image:
                st.subheader("Your output Image")
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                st.image(output_img, use_column_width=True)

    else:

        source_image, input_image = st.columns(2)
        with source_image:
            uploaded_source_images = st.file_uploader("Choose your face you want to blur...", type=["png", "jpg"],
                                                      key=1,
                                                      accept_multiple_files=True)

        with input_image:
            uploaded_verifiing_image = st.file_uploader("Choose your input image...", type=["png", "jpg"], key=2)

        source_image, input_image, output_image = st.columns(3)

        if len(uploaded_source_images) > 0 and uploaded_verifiing_image is not None:

            source_imgs = [Image.open(uploaded_source_image) for uploaded_source_image in uploaded_source_images]
            source_imgs = [cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR) for source_img in source_imgs]

            verifiing_image = Image.open(uploaded_verifiing_image)
            verifiing_image = np.array(verifiing_image)

            output_img = np.copy(verifiing_image)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            if blurred_image == "Source Image":

                for source_img in source_imgs:
                    bbox = getBBoxFromVerifiedFace(source_img, output_img, model)
                    if bbox:
                        imgBlurPart = cv2.GaussianBlur(output_img[bbox[1]:bbox[3],
                                                       bbox[0]:bbox[2]],
                                                       (49, 49), 0)
                        # cv2.rectangle(output_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                        output_img[bbox[1]:bbox[3],
                        bbox[0]:bbox[2]] = imgBlurPart

            elif blurred_image == "Everyone Else":
                for source_img in source_imgs:

                    bbox_source = getBBoxFromVerifiedFace(source_img, output_img, model)
                    bbox = detect_faces(output_img, detector, confidence)

                    if bbox_source is not None and bbox is not None:

                        for box in bbox:

                            iou = get_iou(box, bbox_source)

                            if iou > 0.5:
                                continue

                            imgBlurPart = cv2.GaussianBlur(output_img[box[1]:box[3],
                                                           box[0]:box[2]],
                                                           (49, 49), 0)
                            output_img[box[1]:box[3],
                            box[0]:box[2]] = imgBlurPart

            with source_image:
                st.subheader("Your source Image")
                source_imgs = [cv2.cvtColor(np.array(source_img), cv2.COLOR_BGR2RGB) for source_img in source_imgs]
                st.image(source_imgs, use_column_width=True)

            with input_image:
                st.subheader("Your input Image")
                st.image(verifiing_image, use_column_width=True)

            with output_image:
                st.subheader("Your output Image")
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                st.image(output_img, use_column_width=True)


def blur_video(model, blurred_image, prototxt, caffemodel, confidence=0.5):
    global x_face_recog, y_face_recog, w_face_recog, h_face_recog, path
    os.makedirs("videos", exist_ok=True)
    detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    if blurred_image == "All":

        uploaded_verifiing_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        input_video, output_video = st.columns(2)

        if uploaded_verifiing_video is not None:

            path = os.path.join("videos", uploaded_verifiing_video.name)
            # vid_file = False
            try:
                with open(path, "wb") as f:
                    f.write(uploaded_verifiing_video.getbuffer())
                    vid_file = True
            except PermissionError:
                vid_file = True
                pass

            with input_video:
                st.subheader("Your input Video")
                st.video(uploaded_verifiing_video)

            if vid_file:
                with output_video:
                    st.subheader("Your output Video")

                    cap = cv2.VideoCapture(path)

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    output_path = os.path.join("videos", "result.mp4")

                    writer = create_video_writer(cap, output_path)

                    for _ in stqdm(range(total_frames), colour="green"):

                        success, image = cap.read()
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        if not success:
                            break

                        bbox = detect_faces(image, detector, confidence)
                        if bbox is not None:

                            for box in bbox:
                                imgBlurPart = cv2.GaussianBlur(image[box[1]:box[3],
                                                               box[0]:box[2]],
                                                               (99, 99), 0)
                                image[box[1]:box[3],
                                box[0]:box[2]] = imgBlurPart

                        writer.write(image)

                    cap.release()
                    writer.release()
                    video_file = open(output_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)

    else:

        source_image, input_video = st.columns(2)
        with source_image:
            uploaded_source_images = st.file_uploader("Choose your face you want to blur...", type=["png", "jpg"],
                                                      key=1,
                                                      accept_multiple_files=True)

        with input_video:
            uploaded_verifiing_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        source_image, input_video, output_video = st.columns(3)

        if len(uploaded_source_images) > 0 and uploaded_verifiing_video is not None:

            source_imgs = [Image.open(uploaded_source_image) for uploaded_source_image in uploaded_source_images]
            source_imgs = [cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR) for source_img in source_imgs]

            if uploaded_verifiing_video is not None:

                with source_image:
                    st.subheader("Your source Image")
                    source_imgs = [cv2.cvtColor(np.array(source_img), cv2.COLOR_BGR2RGB) for source_img in source_imgs]
                    st.image(source_imgs, use_column_width=True)

                path = os.path.join("videos", uploaded_verifiing_video.name)
                # vid_file = False
                try:
                    with open(path, "wb") as f:
                        f.write(uploaded_verifiing_video.getbuffer())
                        vid_file = True
                except PermissionError:
                    vid_file = True
                    pass

                with input_video:
                    st.subheader("Your input Video")
                    st.video(uploaded_verifiing_video)

            if vid_file:
                with output_video:
                    st.subheader("Your output Video")
                    cap = cv2.VideoCapture(path)

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    output_path = os.path.join("videos", "result.mp4")

                    writer = create_video_writer(cap, output_path)

                    for _ in stqdm(range(total_frames), colour="green"):

                        success, image = cap.read()
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        if not success:
                            break

                        if blurred_image == "Source Image":

                            for source_img in source_imgs:
                                bbox = getBBoxFromVerifiedFace(source_img, image, model)
                                if bbox:
                                    imgBlurPart = cv2.GaussianBlur(image[bbox[1]:bbox[3],
                                                                   bbox[0]:bbox[2]],
                                                                   (49, 49), 0)
                                    image[bbox[1]:bbox[3],
                                    bbox[0]:bbox[2]] = imgBlurPart

                        elif blurred_image == "Everyone Else":
                            for source_img in source_imgs:

                                bbox_source = getBBoxFromVerifiedFace(source_img, image, model)
                                bbox = detect_faces(image, detector, confidence)

                                if bbox_source is not None and bbox is not None:

                                    for box in bbox:

                                        iou = get_iou(box, bbox_source)

                                        if iou > 0.5:
                                            continue

                                        imgBlurPart = cv2.GaussianBlur(image[box[1]:box[3],
                                                                       box[0]:box[2]],
                                                                       (49, 49), 0)
                                        image[box[1]:box[3],
                                        box[0]:box[2]] = imgBlurPart

                        writer.write(image)

                    cap.release()
                    writer.release()
                    video_file = open(output_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)


def main_page(model_path):
    _ = model_path
    st.markdown("<h1 style='text-align: center; color: white; font-size:300%; text-decoration-line: underline;\
          text-decoration-color: red;  '>Object Detection and Segmentation App</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    p, li {
      font-size: 120%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p font-size:150%> Welcome to our object detection and segmentation app! This app "
        "allows you to perform "
        "object detection and segmentation on images, videos, and webcam streams.</p>", unsafe_allow_html=True)

    # Object detection and segmentation overview
    st.header("Object Detection and Segmentation")
    st.markdown(
        "<p> Object detection is the task of identifying and locating objects in an image or "
        "video. Object segmentation "
        "is the task of identifying and outlining the boundaries of objects in an image or video. </p>",
        unsafe_allow_html=True)

    # Instructions for using the app
    st.header("How to Use the App")
    st.markdown(
        "<p>To use the app, simply upload an image or video, or start your webcam stream. The app will automatically "
        "detect and segment objects in the input data and display the results.</p>", unsafe_allow_html=True)

    # List of app features
    st.header("Features")
    st.markdown("<p>The app offers the following features:</p>", unsafe_allow_html=True)
    st.markdown("<ul >"
                "<li style=font-size:120%> Object detection: Identify and locate objects in images, videos, "
                "and webcam streams.</li>"
                "<li style=font-size:120%>Support for multiple object classes: Detect and segment multiple object "
                "classes simultaneously.</li> "
                "<li style=font-size:120%>Visualize results: Display detected and segmented objects with bounding "
                "boxes and outlines.</li> "
                "</ul>", unsafe_allow_html=True)

    # Advantages of using object detection and segmentation
    st.header("Benefits")
    st.markdown("<p>Object detection and segmentation offer several benefits, including:</p>", unsafe_allow_html=True)
    st.markdown("<ul >"
                "<li style=font-size:120%>Increased efficiency and accuracy in data analysis</li>"
                "<li style=font-size:120%>Enhanced decision-making capabilities</li>"
                "<li style=font-size:120%>Improved automation of tasks boxes and outlines.</li> "
                "</ul>", unsafe_allow_html=True)

    # Real-world applications of object detection and segmentation
    st.header("Applications")
    st.markdown("<p>Object detection and segmentation have a wide range of applications, including:</p>",
                unsafe_allow_html=True)

    st.markdown("<ul >"
                "<li style=font-size:120%>Self-driving cars: Identify and track pedestrians, vehicles, and other "
                "objects in the road environment.</li> "
                "<li style=font-size:120%>Robotics: Enable robots to perceive and interact with their "
                "surroundings.</li> "
                "<li style=font-size:120%>Medical imaging: Identify and analyze medical images, such as X-rays and "
                "MRIs.</li> "
                "<li style=font-size:120%>Satellite imagery analysis: Analyze satellite imagery to identify objects "
                "such as buildings, roads, and vegetation.</li> "
                "<li style=font-size:120%> Security surveillance: Identify and track people and objects in security "
                "footag</li> "
                "</ul>", unsafe_allow_html=True)


def object_detection_page(model_path):
    st.markdown("<h1 style='text-align: center; color: white; font-size:400%; text-decoration-line: underline;\
      text-decoration-color: red;  '>Object Detection</h1>", unsafe_allow_html=True)

    task = "detect"
    model = None
    classes_ids = []
    package = ""

    medium = st.sidebar.radio("Choose your medium", ["Image", "Video", "Webcam"])
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.01)
    model_select = st.sidebar.radio("Choose your Model", ["YOLO", "YOLO-NAS"])
    if model_select == "YOLO":
        package = "ultralytics"
        version = st.sidebar.selectbox("Choose your version",
                                       ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        model = YOLO(os.path.join(model_path, version + ".pt"))
        classes_ids = prepare_classes(model)

    elif model_select == "YOLO-NAS":
        package = "super_gradients"
        device = 'cuda' if cuda.is_available() else "cpu"
        version = st.sidebar.selectbox("Choose your version",
                                       ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
        model = models.get(f"{version}", pretrained_weights="coco")

        model.to(device)

    if medium == "Image":
        handle_image(model, classes_ids, task, package, confidence)

    elif medium == "Video":
        handle_video(model, classes_ids, task, package, confidence)

    elif medium == "Webcam":
        handle_webcam(model, classes_ids, task, package, confidence)


def segmentation_page(model_path):
    st.markdown("<h1 style='text-align: center; color: white; font-size:400%; text-decoration-line: underline;\
          text-decoration-color: red;  '>Segmentation</h1>", unsafe_allow_html=True)

    model = None
    classes_ids = []
    task = "segment"
    package = ""

    medium = st.sidebar.radio("Choose your medium", ["Image", "Video", "Webcam"])
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.01)
    model_select = st.sidebar.radio("Choose your Model", ["YOLO", "SAM"])
    if model_select == "YOLO":
        package = "ultralytics"
        version = st.sidebar.selectbox("Choose your version",
                                       ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"])

        model = YOLO(os.path.join(model_path, version + ".pt"))
        classes_ids = prepare_classes(model)

    elif model_select == "SAM":
        package = "ultralytics"
        version = st.sidebar.selectbox("Choose your version",
                                       ['sam_h', 'sam_l', 'sam_b', 'mobile_sam'])
        model = SAM(os.path.join(model_path, version + ".pt"))
        classes_ids = prepare_classes(model)

    if medium == "Image":
        handle_image(model, classes_ids, task, package, confidence)
    elif medium == "Video":
        handle_video(model, classes_ids, task, package, confidence)
    elif medium == "Webcam":
        handle_webcam(model, classes_ids, task, package, confidence)


def blur_faces_page(model_path):
    st.markdown("<h1 style='text-align: center; color: white; font-size:400%; text-decoration-line: underline;\
              text-decoration-color: red;  '>Blur Faces</h1>", unsafe_allow_html=True)

    medium = st.sidebar.radio("Choose your medium", ["Image", "Video", "Webcam"])

    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace",
              "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
              ]
    model = st.sidebar.radio("Choose your Model", models, index=6)
    blurred_face = st.sidebar.radio("Choose who should be blurred", ["All", "Source Image", "Everyone Else"])

    prototxt = "deploy.prototxt.txt"
    caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, step=0.01,
                                   value=0.5)

    if medium == "Image":
        blur_image(model, blurred_face, prototxt, caffemodel, confidence)
    elif medium == "Video":
        blur_video(model, blurred_face, prototxt, caffemodel, confidence)
    elif medium == "Webcam":
        st.write("still under construction")
        # face_recog_image()


def main():
    # Setting page layout
    st.set_page_config(
        page_title="Kommt auch noch",  # Setting page title
        page_icon="🤖",  # Setting page icon
        layout="wide",  # Setting layout to wide
        initial_sidebar_state="expanded",  # Expanding sidebar by default

    )

    model_path = r"C:\Users\Kleve\PycharmProjects\ComputerVision2\Resources"
    page_names_to_funcs = {
        "Main Page": main_page,
        "Object Detection": object_detection_page,
        "Segmentation": segmentation_page,
        "Blur Faces": blur_faces_page
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page](model_path)


if __name__ == "__main__":
    main()
