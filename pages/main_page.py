# my_app/pages/main_page.py

import streamlit as st

def main_page():
    # If needed: model_path = r"C:\Users\..."
    st.markdown(
        """
        <h1 style='text-align: center; color: white; font-size:300%; 
        text-decoration-line: underline; text-decoration-color: red;'>
            Object Detection and Segmentation App
        </h1>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        p, li {
            font-size: 120%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p font-size:150%> 
        Welcome to our object detection and segmentation app! This app 
        allows you to perform object detection and segmentation 
        on images, videos, and webcam streams.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Object detection and segmentation overview
    st.header("Object Detection and Segmentation")
    st.markdown(
        """
        <p> 
        Object detection is the task of identifying and locating objects in an 
        image or video. Object segmentation is the task of identifying and 
        outlining the boundaries of objects in an image or video. 
        </p>
        """,
        unsafe_allow_html=True
    )

    # etc... (Keep the rest of the big text content you had originally)
    # ...
