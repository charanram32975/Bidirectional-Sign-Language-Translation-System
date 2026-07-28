import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/cnn_model.h5")

# Class labels
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Draw ROI box (where user should place hand)
    x1, y1 = 200, 50
    x2, y2 = 700, 500

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Crop hand region
    roi = frame[y1:y2, x1:x2]

    # Preprocess image
    img = cv2.resize(roi, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    # Predict sign
    prediction = model.predict(img, verbose=0)
    letter = classes[np.argmax(prediction)]

    # Display prediction
    cv2.putText(frame, "Prediction: " + letter,
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2)

    cv2.imshow("Sign Language Recognition", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
