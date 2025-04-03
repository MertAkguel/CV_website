# my_app/modules/detection_utils.py

import random
import cv2
import numpy as np
import streamlit as st

def prepare_classes(model):
    """
    Prepares the class IDs to be used for detection based on user’s selection.
    """
    yolo_classes = list(model.names.values())
    classes = st.sidebar.multiselect(
        'Select your classes',
        ['All'] + sorted(yolo_classes),
        ['All']
    )

    if 'All' not in classes:
        classes_ids = [yolo_classes.index(clas) for clas in classes]
    else:
        classes_ids = list(range(len(yolo_classes)))

    return classes_ids


def predict(chosen_model, img, classes, conf=0.5):
    """
    Helper that calls chosen_model.predict with relevant classes/conf.
    """
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results


def predict_and_detect(chosen_model, img, classes, conf=0.5):
    """
    Runs object detection on an image, draws bounding boxes + labels,
    returns the annotated image & the results.
    """
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                img,
                f"{label}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1
            )
    return img, results


def predict_and_segment(chosen_model, img, classes, conf=0.5):
    """
    Runs segmentation on an image, fills segment masks, returns
    the annotated image & the results.
    """
    results = predict(chosen_model, img, classes, conf=conf)
    # Generate random colors for each class
    colors = [random.choices(range(256), k=3) for _ in classes]

    try:
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                class_index = int(box.cls[0])
                # Map the "class_index" to where it belongs in `classes`
                # (Edge case: if class_index is not in `classes`).
                if class_index in classes:
                    color_idx = classes.index(class_index)
                    cv2.fillPoly(img, points, colors[color_idx])
    except Exception as e:
        print(e)

    return img, results
