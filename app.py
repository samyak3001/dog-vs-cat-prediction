import streamlit as st
import numpy as np
from PIL import Image

st.title("🐶🐱 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Temporary prediction (for deployment)
    if np.random.rand() > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")
