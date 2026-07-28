import cv2
import numpy as np

class ImagePreprocessor:

    def preprocess(self, img):

        img = cv2.resize(img, (64,64))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img / 255.0

        img = img.reshape(1,64,64,1)

        return img
