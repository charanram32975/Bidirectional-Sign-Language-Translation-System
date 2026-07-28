import streamlit as st
import os
from PIL import Image

from streamlit_app import menu

SIGN_FOLDER = "sign_images"

if menu == "Text to Sign":

    st.header("✍️ Text to Sign")

    text = st.text_input("Enter text")

    if st.button("Show Sign"):

        for char in text.upper():

            folder_name = f"{char}-samples"
            folder_path = os.path.join(SIGN_FOLDER, folder_name)

            if os.path.exists(folder_path):

                images = os.listdir(folder_path)

                if len(images) > 0:
                    img_path = os.path.join(folder_path, images[0])
                    st.image(img_path, width=200)
                else:
                    st.write(f"No image inside {folder_name}")

            else:
                st.write(f"No sign image for {char}")
