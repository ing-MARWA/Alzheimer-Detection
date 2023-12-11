import streamlit as st
from tensorflow import keras
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import cv2


st.header("Alzheimer's Disease Prediction")
image = Image.open('image_presentation.jpg')
st.image(image, caption='Brain MRI scans')


# Loading the model 
model2 = load_model("D:\Graduation Project\Alzheimer Detection\model.keras")

file = st.file_uploader("Please upload an MRI image.", type=["jpg", "png"])


def import_and_predict(image_data, model2):
    size = (128, 128) 
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model2.predict(img_reshape)
    return prediction


if file is None:
    st.text("No image file has been uploaded.")
else:
    image = Image.open(file)
    predictions = import_and_predict(image, model2)
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    string = "The patient is predicted to be: " + class_names[np.argmax(predictions)]
    st.success(string)
    st.image(image)
