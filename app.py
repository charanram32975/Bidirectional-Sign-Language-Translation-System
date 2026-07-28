import cv2
import time
from collections import deque

from camera.camera_input import CameraInput
from preprocessing.image_preprocessing import ImagePreprocessor
from preprocessing.hand_detector import HandDetector
from model.gesture_predictor import GesturePredictor
from sign_to_text.sign_to_text import SignToText
from text_to_sign.text_to_sign import TextToSign


# Load labels
with open("data/labels.txt") as f:
    labels = f.read().splitlines()


camera = CameraInput()
hand_detector = HandDetector()
preprocessor = ImagePreprocessor()
predictor = GesturePredictor("cnn_model.h5", labels)
sign_to_text = SignToText()
text_to_sign = TextToSign()


print("1. Sign to Text")
print("2. Text to Sign")

choice = input("Choose mode: ")


# ---------------- SIGN TO TEXT ---------------- #

if choice == "1":

    print("Show sign inside the green box. Press q to quit.")

    recognized_text = ""

    # prediction cooldown
    last_prediction_time = 0

    # stability buffer
    buffer = deque(maxlen=6)

    while True:

        frame = camera.get_frame()

        if frame is None:
            break

        frame = cv2.flip(frame, 1)

        # Fixed region of interest
        x1, y1, x2, y2 = 200, 100, 450, 350

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        roi = frame[y1:y2, x1:x2]

        cv2.imshow("Camera", frame)

        # Detect hand
        hand = hand_detector.detect_and_crop(roi)

        if hand is None:
            continue

        # ignore tiny detections
        if hand.shape[0] < 80 or hand.shape[1] < 80:
            continue


        current_time = time.time()

        # predict every 1.5 seconds
        if current_time - last_prediction_time > 1.5:

            processed = preprocessor.preprocess(hand)

            label, confidence = predictor.predict(processed)

            buffer.append(label)

            # stability check
            if buffer.count(label) >= 5 and confidence > 0.9:

                text = sign_to_text.convert(label)

                recognized_text += text

                print("Recognized:", text)

                buffer.clear()

            last_prediction_time = current_time


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    print("\nFinal Recognized Text:", recognized_text)

    camera.release()
    cv2.destroyAllWindows()


# ---------------- TEXT TO SIGN ---------------- #

elif choice == "2":

    text = input("Enter text: ")

    text_to_sign.show_sign(text)