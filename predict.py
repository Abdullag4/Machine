import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

MODEL_PATH = "shape_model.pth"
IMAGE_SIZE = (64, 64)

# Transform for test image
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Model definition (same as in train.py)
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

def predict_tab():
    st.header("Upload a Shape to Predict")
    test_image = st.file_uploader("Upload a shape image", type=["png", "jpg", "jpeg"])

    if test_image:
        img = Image.open(test_image).convert("RGB")
        st.image(img, caption="Uploaded Test Image", use_column_width=True)

        # Check if model file exists
        try:
            checkpoint = torch.load(MODEL_PATH)

            classes = checkpoint['classes']
            num_classes = len(classes)

            model = SimpleCNN(num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Prepare the image
            input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_index = torch.argmax(outputs, dim=1).item()
                predicted_label = classes[predicted_index]

            st.success(f"🧠 Predicted Shape: **{predicted_label}**")

        except FileNotFoundError:
            st.error("❌ No trained model found. Please train the model first in the 'Train Model' tab.")
