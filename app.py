import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load trained model
model = tf.keras.models.load_model('damage_classification_model.h5')

# Kelas sesuai urutan output model
class_names = ['01-minor', '02-moderate', '03-severe']

# Ukuran gambar input (harus sesuai dengan model)
img_width, img_height = 150, 150

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((img_width, img_height))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambah dimensi batch
    return image

# UI
st.title("🚗 Car Damage Classification")
st.write("Upload an image of a car crash, and the model will predict the severity of damage.")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prediksi jika tombol ditekan
    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"### 🔍 Predicted Severity: **{predicted_class}**")
        st.markdown(f"Confidence Score: **{confidence:.2f}**")
