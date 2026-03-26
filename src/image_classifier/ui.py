from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from src.image_classifier.classifier import ImageNetClassifier


st.set_page_config(
    page_title="Image Classifier",
    page_icon="IC",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the model...")
def load_classifier() -> ImageNetClassifier:
    return ImageNetClassifier()


def render_header() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 2rem 2.25rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.24), transparent 34%),
                linear-gradient(135deg, #f8f5ef 0%, #ded6c6 100%);
            border: 1px solid rgba(20, 33, 61, 0.1);
            box-shadow: 0 18px 48px rgba(20, 33, 61, 0.08);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 3rem;
            line-height: 1;
            letter-spacing: -0.04em;
        }
        .hero p {
            margin-top: 0.9rem;
            max-width: 44rem;
            font-size: 1.05rem;
        }
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(20, 33, 61, 0.08);
        }
        </style>
        <section class="hero">
            <h1>Image Classifier</h1>
            <p>
                Upload a photo and get the model's best guess in seconds.
                This demo uses a pretrained ResNet-50 model trained on ImageNet.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_predictions(predictions: list[tuple[str, float]]) -> None:
    st.subheader("Top predictions")
    for rank, (label, confidence) in enumerate(predictions, start=1):
        st.markdown(
            f"""
            <div class="metric-card">
                <strong>#{rank} {label}</strong><br/>
                Confidence: {confidence:.2%}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence, text=f"{label}: {confidence:.2%}")


def main() -> None:
    render_header()

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a clear photo for the best results.",
        )

        top_k = st.slider("How many predictions to show", min_value=1, max_value=5, value=3)

        st.caption(
            "Supported model: ResNet-50 pretrained on ImageNet. "
            "The first run may take a minute while model weights download."
        )

    with right:
        st.subheader("Preview")
        if uploaded_file is None:
            st.info("Upload an image to preview it and run classification.")
            return

        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        st.image(image, use_container_width=True)

    classifier = load_classifier()
    with st.spinner("Classifying image..."):
        predictions = classifier.predict(image_bytes=image_bytes, top_k=top_k)

    results = [(prediction.label, prediction.confidence) for prediction in predictions]
    render_predictions(results)

    top_prediction = predictions[0]
    st.success(
        f"Best match: {top_prediction.label} "
        f"with {top_prediction.confidence:.2%} confidence."
    )

    with st.expander("How this works"):
        st.write(
            "The app preprocesses your image, runs it through a pretrained "
            "convolutional neural network, then shows the highest-probability labels."
        )
