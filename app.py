"""Streamlit front end: upload a dermatoscopic image, get a prediction and a
Grad-CAM++ explanation.

Research / educational demo only. Not a diagnostic device.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import (
    CKPT_PATH,
    CLASSES,
    FULL_NAME,
    download_checkpoint,
    gradcam_plus_plus,
    load_model,
    overlay,
    predict,
    preprocess,
)

st.set_page_config(page_title="Explainable Skin Lesion Classifier", layout="wide")


def _model_url() -> str:
    """Read the weights URL from Streamlit secrets or the environment."""
    try:
        if "MODEL_URL" in st.secrets:
            return st.secrets["MODEL_URL"]
    except Exception:
        pass
    return os.environ.get("MODEL_URL", "")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    if not CKPT_PATH.exists():
        url = _model_url()
        if not url:
            st.error(
                "No checkpoint found and MODEL_URL is not set. "
                "Add MODEL_URL to your Streamlit secrets."
            )
            st.stop()
        download_checkpoint(url)
    return load_model()


st.title("Explainable Skin Lesion Classification")
st.caption(
    "ResNet-50 trained on HAM10000, explained with Grad-CAM++. "
    "Research and educational use only - this is not a diagnostic tool "
    "and must not be used for medical decisions."
)

uploaded = st.file_uploader(
    "Upload a dermatoscopic image", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload a JPG or PNG image to see the prediction and its explanation.")
    st.stop()

image = Image.open(uploaded)
rgb01, input_tensor = preprocess(image)

model = get_model()
probs = predict(model, input_tensor)
top_idx = int(np.argmax(probs))
top_code = CLASSES[top_idx]

st.subheader(f"Prediction: {FULL_NAME[top_code]}  ({probs[top_idx]:.1%} confidence)")

explain_idx = top_idx
with st.expander("Explain a different class instead"):
    choice = st.selectbox(
        "Class to explain",
        options=list(range(len(CLASSES))),
        index=top_idx,
        format_func=lambda i: f"{FULL_NAME[CLASSES[i]]} ({probs[i]:.1%})",
    )
    explain_idx = int(choice)

with st.spinner("Computing Grad-CAM++ ..."):
    cam = gradcam_plus_plus(model, input_tensor, explain_idx)
    heatmap = overlay(rgb01, cam)

left, right = st.columns(2)
with left:
    st.image(rgb01, caption="Input (224x224)", use_container_width=True)
with right:
    st.image(
        heatmap,
        caption=f"Grad-CAM++ for {FULL_NAME[CLASSES[explain_idx]]}",
        use_container_width=True,
    )

st.subheader("Class probabilities")
table = pd.DataFrame(
    {"Class": [FULL_NAME[c] for c in CLASSES], "Probability": probs}
).sort_values("Probability", ascending=False)
st.bar_chart(table.set_index("Class"))
st.dataframe(
    table.style.format({"Probability": "{:.2%}"}),
    hide_index=True,
    use_container_width=True,
)
