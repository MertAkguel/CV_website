from deepface import DeepFace

def getBBoxFromVerifiedFace(source_img, output_img, model):
    bbox = []
    verification = DeepFace.verify(img1_path=source_img, img2_path=output_img, model_name=model)
    if verification["verified"]:
        x_face_recog, y_face_recog, w_face_recog, h_face_recog = verification["facial_areas"][
            "img2"].values()
        bbox.append(x_face_recog)
        bbox.append(y_face_recog)
        bbox.append(x_face_recog + w_face_recog)
        bbox.append(y_face_recog + h_face_recog)

        return bbox


