import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Title
st.set_page_config(page_title="Cat vs Dog", layout="centered")
st.title("🐶🐱 Cat vs Dog Classifier")

st.write("Upload an image to classify")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize (IMPORTANT → match training size)
    img = img.resize((224, 224))   # change to (160,160) if needed

    # Convert to array
    img_array = np.array(img)

    # Normalize (IMPORTANT)
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Show raw value
    st.write(f"Prediction value: {prediction:.4f}")

    # 🔥 FIXED LABEL LOGIC (IMPORTANT)
    if prediction > 0.5:
        st.success("🐱 Cat")
    else:
        st.success("🐶 Dog") 
