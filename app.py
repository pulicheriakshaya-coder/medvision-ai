import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import pandas as pd

st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩻",
    layout="wide"
)

# ---------- HEADER ----------
st.title("🩻 MedVision AI")
st.subheader("Chest X-ray Abnormality Screening")

st.write(
    "AI-assisted screening of chest X-ray images using a pretrained "
    "deep-learning model."
)

st.warning(
    "⚠️ Educational/Hackathon Prototype — AI scores are not a medical diagnosis."
)

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )
    model.eval()
    return model


# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

    with col2:
        st.info(
            "Image received successfully.\n\n"
            "Click **Analyze X-ray** to run the AI screening model."
        )

    if st.button("🔍 Analyze X-ray", use_container_width=True):

        with st.spinner("Analyzing X-ray with AI model..."):

            model = load_model()

            # Convert image to NumPy
            img = np.array(image)

            # Normalize image
            img = xrv.datasets.normalize(img, 255)

            # Convert RGB to grayscale
            img = img.mean(2)[None, ...]

            # Resize / crop
            transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])

            img = transform(img)

            # Convert to tensor
            img_tensor = torch.from_numpy(img).unsqueeze(0)

            # Model prediction
            with torch.no_grad():
                output = model(img_tensor)[0].cpu().numpy()

            predictions = dict(
                zip(model.pathologies, output)
            )

        st.success("✅ AI screening completed!")

        # ---------- RESULTS ----------
        st.divider()
        st.header("🧠 AI Screening Results")

        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top 3 findings
        top3 = sorted_predictions[:3]

        cols = st.columns(3)

        for col, (finding, score) in zip(cols, top3):

            percentage = float(score) * 100

            with col:
                st.metric(
                    label=finding,
                    value=f"{percentage:.1f}%"
                )

        # ---------- CHART ----------
        st.subheader("📊 Finding Scores")

        chart_data = pd.DataFrame(
            {
                "Finding": [x[0] for x in sorted_predictions[:8]],
                "Score": [
                    max(0, float(x[1]) * 100)
                    for x in sorted_predictions[:8]
                ]
            }
        )

        chart_data = chart_data.set_index("Finding")

        st.bar_chart(chart_data)

        # ---------- TABLE ----------
        st.subheader("📋 Detailed Screening Scores")

        table_data = pd.DataFrame(
            [
                {
                    "Finding": finding,
                    "Screening Score": f"{float(score) * 100:.1f}%"
                }
                for finding, score in sorted_predictions
            ]
        )

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "These values represent model screening outputs. "
            "They do not confirm the presence or absence of disease. "
            "A qualified healthcare professional should interpret medical images."
        )

else:

    st.info(
        "👆 Upload a chest X-ray image above to begin AI-assisted screening."
    )


# ---------- FOOTER ----------
st.divider()

st.caption(
    "MedVision AI • Medical Imaging & Computer Vision • Hackathon Prototype"
)
