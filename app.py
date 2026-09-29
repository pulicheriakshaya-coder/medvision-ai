import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 MedVision AI")
st.subheader("Chest X-ray Abnormality Screening")

st.write(
    "Upload a chest X-ray image for AI-assisted screening. "
    "This prototype is for demonstration purposes and is not a medical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    image_array = np.array(image)

    st.success("X-ray image uploaded successfully!")

    st.info(
        "Image preprocessing pipeline is ready. "
        "A trained medical-imaging model will be connected next."
    )

st.divider()

st.caption(
    "MedVision AI • Hackathon Prototype • For educational demonstration only"
)
