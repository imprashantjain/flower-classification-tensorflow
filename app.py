import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("flower_mobilenet.keras")

class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

st.title("🌸 Flower Classifier")

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:

    image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(224,224)
    )

    st.image(image)

    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )