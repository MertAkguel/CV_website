# my_app/main.py

import streamlit as st
from streamlit_pages.main_page import main_page
from streamlit_pages.object_detection_page import object_detection_page
from streamlit_pages.segmentation_page import segmentation_page
from streamlit_pages.blur_faces_page import blur_faces_page

def main():
    st.set_page_config(
        page_title="Computer Vision Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    page_dict = {
        "Main Page": main_page,
        "Object Detection": object_detection_page,
        "Segmentation": segmentation_page,
        "Blur Faces": blur_faces_page
    }

    selected_page = st.sidebar.selectbox("Select a page", list(page_dict.keys()))
    # The function in page_dict associated with selected_page is then called:
    page_dict[selected_page]()

if __name__ == "__main__":
    main()
