import cv2 as cv
import keras
import numpy as np
from skimage.measure import label, find_contours
from skimage.transform import resize

model = keras.models.load_model('C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/model.h5')


def nonBinaryToBinary(image_non_binary):
    image_b = image_non_binary > 0.6

    return image_b


def getMask(img):
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_resized = resize(img_grey, (224, 224)).reshape((224, 224, 1))
    mask = model.predict(np.asarray([img_resized])).reshape((224, 224))

    return mask


def getContour(img):
    h, w, _ = img.shape
    mask = getMask(img)
    mask_reverted = resize(mask, (h, w))
    binary_image = nonBinaryToBinary(mask_reverted)
    labeled_image = label(binary_image)
    contours = find_contours(labeled_image, 0.8)
    c = sorted(contours, key=lambda x: len(x))[-1]

    extTop = (int(c[c[:, 0].argmin()][1]), int(c[c[:, 0].argmin()][0]))
    extLeft = (int(c[c[:, 1].argmin()][1]), int(c[c[:, 1].argmin()][0]))
    extRight = (int(c[c[:, 1].argmax()][1]), int(c[c[:, 1].argmax()][0]))
    extRightM = (int((extTop[0] + extRight[0]) / 2), int((extTop[1] + extRight[1]) / 2))
    extLeftM = (int((extTop[0] + extLeft[0]) / 2), int((extTop[1] + extLeft[1]) / 2))

    return extTop, extLeft, extLeftM, extRight, extRightM
