import numpy as np
from tensorflow.keras.models import load_model

class GesturePredictor:

    def __init__(self, model_path, labels):

        self.model = load_model(model_path)
        self.labels = labels

    def predict(self, image):

        image = image.reshape(1, 64, 64, 1)

        predictions = self.model.predict(image, verbose=0)

        index = np.argmax(predictions)

        confidence = predictions[0][index]

        label = self.labels[index]

        return label, confidence
