import streamlit as st
import numpy as np
import cv2
import os
from ultralytics import YOLO
from PIL import Image
from stqdm import stqdm
from tqdm import tqdm
from help_functions import create_video_writer


def predict_and_draw(chosen_model, img, conf=0.5):
    results = chosen_model.predict(img, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results


def handle_image(chosen_model, conf=0.5):
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
            result_img, _ = predict_and_draw(chosen_model, new_img, conf)
            st.image(result_img, use_column_width=True)

        # st.download_button("Download output", new_img)


def handle_video(chosen_model, conf=0.5):
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

    input_video, output_video = st.columns(2)

    if uploaded_file is not None:

        path = os.path.join("videos", uploaded_file.name)
        vid_file = False
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
                print("Es klappt jaaaaa")
                print("Aus dem Weg Vegeta, es geht los")
                print(range(total_frames))
                print(tqdm(range(total_frames)))
                for _ in tqdm(range(total_frames), colour="blue"):
                    print("hey")
                    success, image = cap.read()
                    print(success)
                    if not success:
                        print("success")
                        break

                    result_img, _ = predict_and_draw(chosen_model, image, conf)
                    writer.write(result_img)

                cap.release()
                writer.release()
                video_file = open(output_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)


def handle_webcam(chosen_model, conf=0.5):
    st.header("kommt noch Bruder")


if __name__ == "__main__":
    # Setting page layout
    st.set_page_config(
        page_title="Kommt auch noch",  # Setting page title
        page_icon="🤖",  # Setting page icon
        layout="wide",  # Setting layout to wide
        initial_sidebar_state="expanded"  # Expanding sidebar by default
    )

    model = YOLO(r"C:\Users\Kleve\PycharmProjects\ComputerVision2\Resources\yolov8m.pt")

    confidence = 0.5

    st.title("KOmmt noch")

    st.write("Some text is here")

    medium = st.sidebar.radio("Choose your medium", ["Image", "Video", "Webcam"])

    if medium == "Image":
        handle_image(model, confidence)
    elif medium == "Video":
        handle_video(model, confidence)
    elif medium == "Webcam":
        handle_webcam(model, confidence)
