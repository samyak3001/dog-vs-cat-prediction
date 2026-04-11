import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import gdown

# Download model
url = "https://drive.google.com/uc?id=15HYsAWvbom5uwc8F3WtIssfgfC4M_FjA"
gdown.download(url, "model.keras", quiet=False)

# Load model
model = load_model("model.keras")

# UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((160, 160))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.success("🐶 Dog")
        st.write(f"Confidence: {prediction:.2f}")
    else:
        st.success("🐱 Cat")
        st.write(f"Confidence: {1 - prediction:.2f}")
