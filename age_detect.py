from PIL import Image
import numpy as np
import cv2 as cv
from mtcnn import MTCNN

detector = MTCNN()


def extract_face(image):
    detections = detector.detect_faces(image)
    if len(detections) == 0:
        return 0, 0, 0, 0, 0
    for face in detections:
        x, y, w, h = face["box"]
        cropped_image = image[y:(y + h), x:(x + w)]
        cropped_image_res = np.array(Image.fromarray(cropped_image).resize([224, 224]))

        return cropped_image_res, x, y, w, h


def age_detection(path, model):
    image = cv.imread(path)
    face, x, y, w, h = extract_face(image)
    if face.any() is None:
        return None, 0

    age_prediction = model.predict(face.reshape(1, 224, 224, 3))[0][0]

    return image, int(age_prediction)
