import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶🐱",
    layout="centered"
)

# Load model
model = tf.keras.models.load_model(
    "cat_dog_model.h5",
    compile=False
)

# Title
st.title("🐶🐱 Cat vs Dog Classifier")

st.write("Upload an image to classify it as a Cat or Dog.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize to training size
    img = img.resize((160, 160))

    # Convert to NumPy array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # Show raw prediction
    st.write(f"Prediction value: {prediction:.4f}")

    # Classification
    if prediction > 0.5:
        confidence = prediction * 100
        st.success(
            f"🐶 Dog — Confidence: {confidence:.2f}%"
        )
    else:
        confidence = (1 - prediction) * 100
        st.success(
            f"🐱 Cat — Confidence: {confidence:.2f}%"
        )
