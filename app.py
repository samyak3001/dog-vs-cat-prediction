import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
import os
import tflite_runtime.interpreter as tflite

# Download model WITHOUT gdown
MODEL_URL = "https://drive.google.com/uc?export=download&id=1nn78IVOV55kUdI96nUjNjqrfu-BP---c"
MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# UI
st.title("🐶🐱 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)

    img = img.resize((160, 160))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")
