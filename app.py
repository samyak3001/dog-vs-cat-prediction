import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PawVision AI",
    page_icon="🐾",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f1117 0%,
            #171522 50%,
            #10131a 100%
        );
    }

    /* Main container */
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #a1a1aa;
        font-size: 16px;
        margin-bottom: 30px;
    }

    /* Upload box */
    .upload-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .upload-description {
        color: #a1a1aa;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Result title */
    .result-title {
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        margin-top: 20px;
    }

    .result-confidence {
        text-align: center;
        color: #a1a1aa;
        font-size: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #71717a;
        font-size: 13px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🐾 PawVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Cat & Dog Image Classification</div>',
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "cat_dog_model.h5",
        compile=False
    )


model = load_model()

# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="upload-title">📸 Upload an Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-description">'
    'Upload a clear image of a cat or dog to classify it.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # --------------------------------------------------------
    # Display smaller image
    # --------------------------------------------------------

    st.image(
        image,
        caption="Uploaded Image",
        width=450
    )

    # --------------------------------------------------------
    # Prepare image for model
    # --------------------------------------------------------

    resized_image = image.resize((160, 160))

    image_array = np.array(resized_image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # --------------------------------------------------------
    # Determine class
    # --------------------------------------------------------

    if prediction > 0.5:

        animal = "Dog"
        emoji = "🐶"
        confidence = prediction * 100

    else:

        animal = "Cat"
        emoji = "🐱"
        confidence = (1 - prediction) * 100

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        f'<div class="result-title">'
        f'{emoji} {animal}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-confidence">'
        f'Confidence: <b>{confidence:.2f}%</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Confidence progress bar
    st.progress(
        min(int(confidence), 100)
    )

    # Prediction score
    st.caption(
        f"Prediction score: {prediction:.4f}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    if animal == "Dog":

        st.success(
            f"🐶 The model predicts this image is a Dog "
            f"with {confidence:.2f}% confidence."
        )

    else:

        st.info(
            f"🐱 The model predicts this image is a Cat "
            f"with {confidence:.2f}% confidence."
        )

# ============================================================
# ABOUT SECTION
# ============================================================

st.divider()

with st.expander("ℹ️ About PawVision AI"):

    st.write(
        """
        **PawVision AI** is a deep-learning based image
        classification application that identifies whether
        an uploaded image contains a Cat or Dog.

        **How it works:**

        1. Upload an image.
        2. The image is converted to RGB.
        3. The image is resized to 160 × 160 pixels.
        4. Pixel values are normalized.
        5. The trained neural network processes the image.
        6. The application predicts Cat or Dog.
        7. The confidence score is displayed.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '🐾 PawVision AI • Deep Learning Image Classifier'
    '</div>',
    unsafe_allow_html=True
)
