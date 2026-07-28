import cv2
import numpy as np

class HandDetector:

    def detect_and_crop(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) < 5000:   # ignore small noise
            return None

        x, y, w, h = cv2.boundingRect(c)

        hand = frame[y:y+h, x:x+w]

        return hand
