import streamlit as st
import numpy as np
from PIL import Image
import gdown
import tflite_runtime.interpreter as tflite

# Download model from Google Drive
url = "https://drive.google.com/uc?id=1nn78IVOV55kUdI96nUjNjqrfu-BP---c"
gdown.download(url, "model.tflite", quiet=False)

# Load model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

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
