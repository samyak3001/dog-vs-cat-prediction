import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PawVision AI",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(236, 72, 153, 0.10),
                transparent 30%
            ),
            #0f1117;
        color: #ffffff;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 850px;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 25px 10px 15px 10px;
    }

    .logo {
        font-size: 55px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        background: linear-gradient(
            90deg,
            #8b5cf6,
            #ec4899,
            #f59e0b
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #a1a1aa;
        font-size: 16px;
        margin-top: 10px;
    }

    /* Upload card */
    .upload-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 20px;
        padding: 25px;
        margin-top: 25px;
        box-shadow: 0 15px 45px rgba(0,0,0,0.25);
    }

    /* Section title */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Result card */
    .result-card {
        margin-top: 25px;
        padding: 25px;
        border-radius: 20px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        text-align: center;
    }

    .result-animal {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .result-name {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .confidence {
        color: #a1a1aa;
        font-size: 15px;
    }

    /* Confidence bar */
    .bar-container {
        width: 100%;
        height: 10px;
        background: #27272a;
        border-radius: 20px;
        margin-top: 15px;
        overflow: hidden;
    }

    .bar {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(
            90deg,
            #8b5cf6,
            #ec4899
        );
    }

    /* Prediction value */
    .prediction-value {
        margin-top: 15px;
        color: #71717a;
        font-size: 13px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #52525b;
        font-size: 13px;
        margin-top: 45px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.025);
        border-radius: 15px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px;
        font-weight: 700;
    }

</style>
""", unsafe_allow_html=True)


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
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <div class="logo">🐾</div>

    <div class="hero-title">
        PawVision AI
    </div>

    <div class="hero-subtitle">
        Intelligent Cat & Dog Image Classification
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown("""
<div class="upload-card">

<div class="section-title">
📸 Upload an Image
</div>

<p style="color:#a1a1aa;">
Upload a clear image of a cat or dog and let PawVision AI identify it.
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    # Display uploaded image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    resized_img = img.resize((160, 160))

    # Convert to array
    img_array = np.array(resized_img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # Determine class
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

    st.markdown(f"""
    <div class="result-card">

        <div class="result-animal">
            {emoji}
        </div>

        <div class="result-name">
            {animal}
        </div>

        <div class="confidence">
            Model confidence: <b>{confidence:.2f}%</b>
        </div>

        <div class="bar-container">

            <div
                class="bar"
                style="width:{confidence:.2f}%"
            ></div>

        </div>

        <div class="prediction-value">
            Prediction score: {prediction:.4f}
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# INFORMATION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("ℹ️ About PawVision AI"):

    st.write("""
    PawVision AI is a deep-learning based image classifier
    designed to distinguish between cats and dogs.

    **How it works:**

    1. Upload an image.
    2. The image is resized to 160 × 160 pixels.
    3. The image is normalized.
    4. The trained neural network analyzes the image.
    5. The system predicts Cat or Dog.
    6. A confidence score is displayed.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🐾 PawVision AI · Powered by Deep Learning

</div>
""", unsafe_allow_html=True)
