import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

# Global model file path
MODEL_PATH = "shape_model.pth"

# Resize shape for input
IMAGE_SIZE = (64, 64)

# Transform image → tensor
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),  # Converts [0-255] to [0.0-1.0]
])

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_tab():
    st.header("Upload and Label Shape Images for Training")

    train_images = st.file_uploader(
        "Upload images of shapes (e.g., circle, triangle, square)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    labeled_data = []

    if train_images:
        st.success(f"{len(train_images)} image(s) uploaded.")
        st.write("Label each image below:")

        for i, img_file in enumerate(train_images):
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption=f"Image {i+1}", use_column_width=True)
            label = st.text_input(f"Label for image {i+1}:", key=f"label_{i}")
            if label:
                labeled_data.append((img, label))

    if labeled_data and st.button("🚀 Train Model"):
        st.info("Training the model... please wait ⏳")

        # Preprocess images
        X = []
        y = []
        for img, label in labeled_data:
            img_tensor = transform(img)
            X.append(img_tensor)
            y.append(label)

        X = torch.stack(X)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        num_classes = len(label_encoder.classes_)
        model = SimpleCNN(num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        model.train()
        for epoch in range(30):  # Small number of epochs for quick training
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Save model and labels
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': label_encoder.classes_
        }, MODEL_PATH)

        st.success("✅ Model trained and saved successfully!")
