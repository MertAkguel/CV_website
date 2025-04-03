# my_app/pages/blur_faces_page.py

import streamlit as st

from modules.face_blur_utils import blur_image, blur_video

def blur_faces_page():
    st.markdown("""
    <h1 style='text-align: center; color: white; font-size:400%;
    text-decoration-line: underline; text-decoration-color: red;'>
        Blur Faces
    </h1>
    """, unsafe_allow_html=True)

    medium = st.sidebar.radio("Choose your medium", ["Image", "Video", "Webcam"])
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace",
              "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

    model = st.sidebar.radio("Choose your Model", models, index=6)
    blurred_face = st.sidebar.radio(
        "Choose who should be blurred",
        ["All", "Source Image", "Everyone Else"]
    )

    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, step=0.01, value=0.5)

    # Example paths - update with your own paths if needed
    prototxt = "deploy.prototxt.txt"
    caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"

    if medium == "Image":
        blur_image(model, blurred_face, prototxt, caffemodel, confidence)
    elif medium == "Video":
        blur_video(model, blurred_face, prototxt, caffemodel, confidence)
    elif medium == "Webcam":
        st.write("Still under construction!")
