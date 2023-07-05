import io

import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow import keras
from PIL import Image, ImageOps

class CAPDDetection():
    def __init__(self) -> None:
        with tf.device('/cpu:0'):
            self.capd_model = keras.models.load_model("./model/best_model.h5", compile=False)
        with open("./model/labels.txt") as file:
            self.capd_labels = file.read().splitlines()
        
    def load_image(self):
        uploaded_file = st.file_uploader(label='Pick an effluent dialysate image')
        if uploaded_file is not None:
            img_data = uploaded_file.getvalue()
            st.image(img_data)
            return Image.open(io.BytesIO(img_data))

    def predict(self, image):
        size = (224, 224)
        resized_image = ImageOps.fit(image, size, Image.ANTIALIAS)
        
        image_array = np.asarray(resized_image)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = image_array
        preds = self.capd_model.predict(data).flatten()

        preds_proba = tf.nn.sigmoid(preds)
        preds = tf.where(preds_proba < 0.5, 0, 1)
        preds_result = preds.numpy().tolist()

        labels_list = list(self.capd_labels)

        label = labels_list[int(preds_result[0])]
        label_proba = preds_proba.numpy().tolist()

        return label, label_proba


    
def main():
    st.title('CAPD Upload Demo')
    capd_detection = CAPDDetection()
    image = capd_detection.load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        label, label_proba = capd_detection.predict(image)
        st.write({"label": str(label),
                  "accuracy": str(label_proba)
                  })

if __name__ == '__main__':
    main()