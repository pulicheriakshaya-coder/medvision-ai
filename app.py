import streamlit as st
from PIL import Image

st.set_page_config(page_title="MedVision AI", layout="centered")
st.title("MedVision AI")
st.caption("Medical image analysis demo")

def predict(image: Image.Image) -> dict:
    # TODO: replace with real model inference
    return {"label": "Placeholder", "confidence": 0.0}

uploaded = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    result = predict(img)
    st.write(f"Prediction: **{result['label']}** ({result['confidence']:.2f})")
