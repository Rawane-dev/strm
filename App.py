import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱", layout='wide')

IMG_SIZE = 224

model = tf.keras.models.load_model(
    "animal_classifier.keras",
    compile=False
)

class_names = ["cat", "dog"]

st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and the model will predict the class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, width="stretch")

  
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Predict"):
        predictions = model.predict(image_array)
        confidence = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]

        if confidence < 0.7:
            st.warning("⚠️ I'm not confident about this image. Try another one.")
        else:
            st.success(f"### Prediction: {predicted_class}")
            

        cat_prob = predictions[0][0] * 100
        dog_prob = predictions[0][1] * 100

        st.write(f"Cat: {cat_prob:.0f}%")
        st.write(f"Dog: {dog_prob:.0f}%")
        
