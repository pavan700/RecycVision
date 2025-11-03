import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# =====================
# LOAD MODEL
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
model.load_state_dict(torch.load("recycvision_model.pth", map_location=device))
model.eval()

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['Non-Recyclable', 'Recyclable']

# =====================
# STREAMLIT UI
# =====================
st.title("♻️ RecycVision — Smart Waste Classification")
st.write("Upload an image to check whether it’s recyclable or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item() * 100

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f}% confidence)")
