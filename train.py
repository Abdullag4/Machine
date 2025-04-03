import streamlit as st
from PIL import Image
import numpy as np

def train_tab():
    st.header("Upload and Label Shape Images for Training")

    # Upload multiple images
    train_images = st.file_uploader(
        "Upload images of shapes (e.g., circle, triangle, square)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if train_images:
        st.success(f"{len(train_images)} image(s) uploaded.")
        st.write("Label each image below:")

        labeled_data = []

        for i, img_file in enumerate(train_images):
            # Show image
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption=f"Image {i+1}", use_column_width=True)

            # Input label
            label = st.text_input(f"Label for image {i+1}:", key=f"label_{i}")

            # Store image and label
            if label:
                labeled_data.append((img, label))

        if labeled_data:
            st.success(f"{len(labeled_data)} image(s) labeled.")
            if st.button("📚 Prepare for Training"):
                st.info("✅ Data is ready! You can now train the model. (Training logic coming next)")
