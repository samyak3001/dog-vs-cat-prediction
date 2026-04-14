import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Title
st.title("🐶🐱 Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (VERY IMPORTANT)
    img = img.resize((224, 224))   # change to 160 if your model used 160
    img = np.array(img)

    # Normalize (IMPORTANT)
    img = img / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    # Output
    st.write(f"Raw Prediction: {prediction:.4f}")

    if prediction > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")
