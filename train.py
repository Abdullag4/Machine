import streamlit as st
from PIL import Image

def train_tab():
    st.header("Upload Shape Images for Training")
    train_images = st.file_uploader("Upload images of shapes (e.g., circle, triangle)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if train_images:
        st.success(f"{len(train_images)} image(s) uploaded for training.")
        for img_file in train_images:
            img = Image.open(img_file)
            st.image(img, caption=img_file.name, use_column_width=True)

    st.warning("🚧 Training functionality will be added soon.")
