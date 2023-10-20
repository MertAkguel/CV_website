import streamlit as st
import random
import numpy as np
import cv2
import os
from ultralytics import YOLO, SAM
from PIL import Image
from stqdm import stqdm
from torch import cuda
from help_functions import create_video_writer
from super_gradients.training import models


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

    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            # cv2.polylines(img, points, True, (255, 0, 0), 1)
            color_number = classes.index(int(box.cls[0]))
            cv2.fillPoly(img, points, colors[color_number])

    return img, results


def handle_image(chosen_model, classes, task, package, conf=0.5):
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

            elif package == "super_gradients":

                chosen_model.predict(new_img, conf=conf).save("images")
                result_img = cv2.imread(os.path.join("images", os.listdir("images")[0]))
                st.image(result_img, use_column_width=True)

        # st.download_button("Download output", new_img)


def handle_video(chosen_model, classes, task, package, conf=0.5):
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
                        print("success")
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


def handle_webcam(chosen_model, classes, task, package, conf=0.5):
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


def main_page(model_path):
    _ = model_path
    st.title("KOmmt noch")
    st.write("Some text is here")


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
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page](model_path)


if __name__ == "__main__":
    main()
