import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv

st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 MedVision AI")
st.subheader("Chest X-ray Abnormality Screening")

st.warning(
    "This is an AI-assisted screening prototype for educational/hackathon use. "
    "It is NOT a medical diagnosis."
)

@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )
    model.eval()
    return model


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

    if st.button("🔍 Analyze X-ray"):

        with st.spinner("Loading AI model and analyzing image..."):

            model = load_model()

            # Convert image to numpy
            img = np.array(image)

            # Normalize image to TorchXRayVision range
            img = xrv.datasets.normalize(img, 255)

            # Convert RGB to single grayscale channel
            img = img.mean(2)[None, ...]

            # Resize and center crop
            transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])

            img = transform(img)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(img).unsqueeze(0)

            # Run model
            with torch.no_grad():
                output = model(img_tensor)[0].cpu().numpy()

            predictions = dict(
                zip(model.pathologies, output)
            )

        st.success("Analysis completed!")

        st.subheader("🧠 AI Screening Results")

        # Sort findings by score
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Display top findings
        for finding, score in sorted_predictions[:8]:

            percentage = float(score) * 100

            st.write(
                f"**{finding}** — {percentage:.1f}%"
            )

            st.progress(
                min(max(float(score), 0.0), 1.0)
            )

        st.info(
            "These scores are model outputs and should not be interpreted "
            "as a confirmed diagnosis. Clinical evaluation is required."
        )

st.divider()

st.caption(
    "MedVision AI • Medical Imaging & Computer Vision • Hackathon Prototype"
)
