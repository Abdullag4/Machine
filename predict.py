import streamlit as st
from PIL import Image

def predict_tab():
    st.header("Upload a Shape to Predict")
    test_image = st.file_uploader("Upload a shape image", type=["png", "jpg", "jpeg"])

    if test_image:
        img = Image.open(test_image)
        st.image(img, caption="Test Image", use_column_width=True)

        st.warning("🚧 Prediction functionality will be added soon.")
