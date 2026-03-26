from __future__ import annotations

from io import BytesIO
import json

import pandas as pd
import streamlit as st
from PIL import Image

from src.image_classifier.classifier import ImageNetClassifier


st.set_page_config(
    page_title="Vision Insight",
    page_icon="VI",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the model...")
def load_classifier() -> ImageNetClassifier:
    return ImageNetClassifier()


def render_header() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #14213d;
            --teal: #0f766e;
            --sand: #f4f1ea;
            --cream: #fbf8f2;
            --gold: #d8a94d;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        .hero {
            padding: 2.5rem 2.7rem;
            border-radius: 30px;
            background:
                radial-gradient(circle at 0% 0%, rgba(216, 169, 77, 0.34), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(15, 118, 110, 0.28), transparent 32%),
                linear-gradient(135deg, #fffaf0 0%, #e8dfd2 52%, #d9ecdf 100%);
            border: 1px solid rgba(20, 33, 61, 0.08);
            box-shadow: 0 22px 60px rgba(20, 33, 61, 0.08);
            margin-bottom: 1.5rem;
        }
        .eyebrow {
            display: inline-flex;
            margin-bottom: 1rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(20, 33, 61, 0.08);
            font-size: 0.84rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .hero h1 {
            margin: 0;
            font-size: 3.5rem;
            line-height: 0.96;
            letter-spacing: -0.04em;
            max-width: 12ch;
        }
        .hero p {
            margin-top: 1rem;
            max-width: 42rem;
            font-size: 1.06rem;
        }
        .hero-stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1.25rem;
        }
        .hero-stat {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(20, 33, 61, 0.08);
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1.4rem;
        }
        .feature-card {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid rgba(20, 33, 61, 0.08);
        }
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(20, 33, 61, 0.08);
            margin-bottom: 0.7rem;
        }
        .callout {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(15, 118, 110, 0.16);
            margin-bottom: 1rem;
        }
        .section-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.6rem 0 1.2rem;
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.72));
            border: 1px solid rgba(20, 33, 61, 0.08);
            box-shadow: 0 10px 30px rgba(20, 33, 61, 0.05);
        }
        .trust-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.4rem;
        }
        .trust-item {
            padding: 0.85rem 0.95rem;
            border-radius: 16px;
            background: rgba(20, 33, 61, 0.04);
            border: 1px solid rgba(20, 33, 61, 0.08);
            font-size: 0.95rem;
        }
        </style>
        <section class="hero">
            <div class="eyebrow">Customer Ready Vision Tool</div>
            <h1>Understand what is in an image, instantly.</h1>
            <p>
                Vision Insight helps teams quickly identify the most likely
                objects or scenes inside an image. Upload a file or capture a
                photo, review ranked predictions, and export results in a format
                your team can use.
            </p>
            <div class="hero-stats">
                <div class="hero-stat"><strong>Fast review</strong><br/>Get ranked predictions in seconds.</div>
                <div class="hero-stat"><strong>Simple workflow</strong><br/>Upload, review, compare, and export.</div>
                <div class="hero-stat"><strong>Built for common imagery</strong><br/>Best for products, objects, animals, vehicles, and scenes.</div>
            </div>
            <div class="feature-grid">
                <div class="feature-card"><strong>Designed for business users</strong><br/>Clear wording, guided steps, and confidence summaries instead of raw model output.</div>
                <div class="feature-card"><strong>Decision-friendly results</strong><br/>See top matches, confidence levels, and structured prediction tables for faster review.</div>
                <div class="feature-card"><strong>Ready to share</strong><br/>Download results as JSON and keep a short on-screen history of recent checks.</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_customer_sections() -> None:
    st.markdown(
        """
        <div class="trust-strip">
            <div class="trust-item"><strong>Easy to use</strong><br/>No technical setup required for end users.</div>
            <div class="trust-item"><strong>Transparent output</strong><br/>Confidence scores help teams judge certainty.</div>
            <div class="trust-item"><strong>Flexible input</strong><br/>Works with uploaded files and live camera capture.</div>
            <div class="trust-item"><strong>Exportable results</strong><br/>Prediction data can be shared or stored.</div>
        </div>
        <div class="section-grid">
            <div class="section-card">
                <strong>Who this is for</strong><br/>
                Customer support teams, operations teams, retail demos, internal
                QA workflows, and anyone who needs a fast first-pass visual check.
            </div>
            <div class="section-card">
                <strong>What it does well</strong><br/>
                Recognizes broad categories such as everyday objects, food,
                animals, transport, plants, and outdoor scenes.
            </div>
            <div class="section-card">
                <strong>How to use it</strong><br/>
                Add an image, review the top predictions, compare confidence,
                and export the result when you need to share or log it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_tone(confidence: float) -> str:
    if confidence >= 0.75:
        return "High confidence"
    if confidence >= 0.45:
        return "Moderate confidence"
    return "Low confidence"


def build_result_payload(
    image: Image.Image,
    predictions: list,
    source_label: str,
) -> dict[str, object]:
    return {
        "source": source_label,
        "image_width": image.width,
        "image_height": image.height,
        "predictions": [
            {
                "label": prediction.label,
                "display_label": prediction.display_label,
                "confidence": round(prediction.confidence, 6),
            }
            for prediction in predictions
        ],
    }


def render_predictions(predictions: list) -> pd.DataFrame:
    st.subheader("Top predictions")
    table = pd.DataFrame(
        [
            {
                "Rank": rank,
                "Label": prediction.display_label,
                "Confidence": prediction.confidence,
                "Confidence %": f"{prediction.confidence:.2%}",
            }
            for rank, prediction in enumerate(predictions, start=1)
        ]
    )

    for rank, prediction in enumerate(predictions, start=1):
        st.markdown(
            f"""
            <div class="metric-card">
                <strong>#{rank} {prediction.display_label}</strong><br/>
                Confidence: {prediction.confidence:.2%}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(
            prediction.confidence,
            text=f"{prediction.display_label}: {prediction.confidence:.2%}",
        )

    return table


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="callout">
            <strong>Start with a clear image.</strong><br/>
            Product photos, packaged items, pets, cars, flowers, food, and everyday objects usually give the strongest results.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload an image or use the camera to begin analysis.")


def render_sidebar() -> tuple[int, bytes | None, str | None]:
    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Predictions to show", min_value=1, max_value=5, value=3)
        input_mode = st.radio("Input type", ["Upload image", "Use camera"], index=0)
        st.caption(
            "Model: MobileNetV3-Large trained on ImageNet. First run may take a moment."
        )
        st.divider()
        st.subheader("Guidance")
        st.write(
            "This experience is built for broad visual recognition. It is ideal for general image understanding, not custom brand- or company-specific classes."
        )
        st.write(
            "For the best output, use one main subject, good lighting, and a sharp image."
        )

        if input_mode == "Upload image":
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Use a clear photo for the best results.",
            )
            if uploaded_file is None:
                return top_k, None, None
            return top_k, uploaded_file.getvalue(), "upload"

        camera_file = st.camera_input("Take a picture")
        if camera_file is None:
            return top_k, None, None
        return top_k, camera_file.getvalue(), "camera"


def update_history(source_label: str, image: Image.Image, prediction_label: str, confidence: float) -> None:
    history = st.session_state.setdefault("prediction_history", [])
    history.insert(
        0,
        {
            "Source": source_label,
            "Image Size": f"{image.width} x {image.height}",
            "Best Match": prediction_label,
            "Confidence": f"{confidence:.2%}",
        },
    )
    del history[5:]


def render_history() -> None:
    history = st.session_state.get("prediction_history", [])
    if not history:
        return

    st.subheader("Recent predictions")
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def main() -> None:
    render_header()
    render_customer_sections()
    top_k, image_bytes, source_label = render_sidebar()

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("Analyze an image")
        st.write(
            "Upload a file or take a picture to receive ranked predictions, confidence indicators, and downloadable result data."
        )
        st.caption(
            "Recommended for customer-facing workflows where teams need a quick, readable first-pass classification."
        )

    with right:
        st.subheader("Preview")
        if image_bytes is None or source_label is None:
            render_empty_state()
            return

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption(f"Source: {source_label} | Size: {image.width} x {image.height}")

    classifier = load_classifier()
    with st.spinner("Classifying image..."):
        predictions = classifier.predict(image_bytes=image_bytes, top_k=top_k)

    top_prediction = predictions[0]
    update_history(
        source_label=source_label.title(),
        image=image,
        prediction_label=top_prediction.display_label,
        confidence=top_prediction.confidence,
    )

    overview_a, overview_b, overview_c = st.columns(3)
    overview_a.metric("Best match", top_prediction.display_label)
    overview_b.metric("Confidence", f"{top_prediction.confidence:.2%}")
    overview_c.metric("Certainty", prediction_tone(top_prediction.confidence))

    st.success(
        f"Most likely result: {top_prediction.display_label} "
        f"at {top_prediction.confidence:.2%} confidence."
    )

    results_table = render_predictions(predictions)

    chart_col, data_col = st.columns([1.15, 1], gap="large")
    with chart_col:
        st.subheader("Confidence chart")
        st.bar_chart(results_table.set_index("Label")["Confidence"], color="#0F766E")
    with data_col:
        st.subheader("Result data")
        st.dataframe(
            results_table[["Rank", "Label", "Confidence %"]],
            use_container_width=True,
            hide_index=True,
        )

        payload = build_result_payload(image, predictions, source_label.title())
        st.download_button(
            "Download predictions as JSON",
            data=json.dumps(payload, indent=2),
            file_name="predictions.json",
            mime="application/json",
            use_container_width=True,
        )

    render_history()

    with st.expander("How this works"):
        st.write(
            "The app prepares your image, evaluates it with a pretrained computer-vision model, and returns the most likely ImageNet categories ranked by confidence."
        )
        st.write(
            "Because this is a general-purpose model, results are strongest for common objects, animals, vehicles, plants, food, and scenes rather than niche custom categories."
        )
        st.write(
            "For customer use, treat the output as a fast recommendation layer. If your workflow needs business-specific labels, the next step would be a custom-trained model."
        )
