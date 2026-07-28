from ultralytics import YOLO
import cv2

class YOLOHandDetector:

    def __init__(self):

        # pretrained hand detection model
        self.model = YOLO("yolov8n.pt")

    def detect_hand(self, frame):

        results = self.model(frame, conf=0.5)

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                hand = frame[y1:y2, x1:x2]

                return hand

        return None
