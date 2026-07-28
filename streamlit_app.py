import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from gtts import gTTS

st.title(" Sign Language Recognition System")

# Load CNN model
model = load_model("model/cnn_model.h5")

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del","nothing","space"]

menu = st.sidebar.selectbox(
    "Choose Feature",
    ["Text to Sign", "Sign to Text"]
)
# ---------------------------
# TEXT TO SIGN
# ---------------------------

if menu == "Text to Sign":

    st.header("Text to Sign")

    DATASET_PATH = "dataset/asl_alphabet_train"

    text = st.text_input("Enter text")

    if st.button("Show Sign"):

        # Convert text to voice
        if text.strip() != "":
            tts = gTTS(text=text, lang="en")
            tts.save("voice.mp3")
            st.audio("voice.mp3")

        # Show sign images
        for char in text.upper():

            if char == " ":
                st.write("SPACE")
                continue

            folder_path = os.path.join(DATASET_PATH, char)

            if os.path.exists(folder_path):

                images = os.listdir(folder_path)

                if len(images) > 0:
                    img_path = os.path.join(folder_path, images[0])
                    st.image(img_path, width=200)

            else:
                st.write(f"No sign image for {char}")

# ---------------------------
# SIGN TO TEXT
# ---------------------------

elif menu == "Sign to Text":

    st.header("Sign to Text (Webcam)")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    text_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    recognized_text = ""

    while run:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ROI box
        x1, y1 = 150, 50
        x2, y2 = 750, 550

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        roi = frame[y1:y2, x1:x2]

        img = cv2.resize(roi, (64,64))
        img = img / 255.0
        img = np.reshape(img, (1,64,64,3))

        prediction = model.predict(img, verbose=0)

        letter = classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        if confidence > 0.85 and letter not in ["nothing"]:

            if letter == "space":
                recognized_text += " "

            elif letter == "del":
                recognized_text = recognized_text[:-1]

            else:
                recognized_text += letter

        cv2.putText(frame,
                    f"{letter} ({confidence:.2f})",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2)

        FRAME_WINDOW.image(frame, channels="BGR")

        text_placeholder.write("Detected Text: " + recognized_text)

    cap.release()