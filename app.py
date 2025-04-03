import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Shape Classifier", layout="centered")

st.title("🔵🟥 Shape Classifier")

# Create tabs
tab1, tab2 = st.tabs(["📚 Train Model", "🔍 Test Model"])

# 📚 Tab 1: Train Model
with tab1:
    st.header("Upload Shape Images for Training")
    train_images = st.file_uploader("Upload images of shapes (e.g., circle, triangle)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if train_images:
        st.success(f"{len(train_images)} image(s) uploaded for training.")
        for img_file in train_images:
            img = Image.open(img_file)
            st.image(img, caption=img_file.name, use_column_width=True)

# 🔍 Tab 2: Test Model
with tab2:
    st.header("Upload a Shape to Predict")
    test_image = st.file_uploader("Upload a shape image", type=["png", "jpg", "jpeg"])

    if test_image:
        img = Image.open(test_image)
        st.image(img, caption="Test Image", use_column_width=True)

        st.warning("🚧 Model training not implemented yet. We'll add prediction soon!")
