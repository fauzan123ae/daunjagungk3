import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Inisialisasi path model (model harus ada di folder project)
MODEL_PATH = 'corn_leaf_disease_model.pth'

CLASS_NAMES = ['blight', 'common_rust', 'gray_leaf_spot', 'healthy']
IMG_SIZE = 224  # Sesuai input size ResNet

# Cek apakah model ada
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model tidak ditemukan di path: {MODEL_PATH}")
    st.stop()

# Load Model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
        label = CLASS_NAMES[predicted.item()]
        return label, confidence

# Streamlit UI
st.title("🌽 Deteksi Penyakit Daun Jagung ")
st.write("Upload gambar daun jagung, lalu sistem akan mendeteksi penyakit berdasarkan model yang dilatih.")

uploaded_file = st.file_uploader("📤 Upload Gambar Daun Jagung", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Gambar yang Diupload", use_column_width=True)

    if st.button("🔍 Prediksi Gambar"):
        with st.spinner("Sedang memproses..."):
            label, confidence = predict_image(image, model)
            st.success(f"✅ Hasil Prediksi: **{label}** ({confidence * 100:.2f}%)")
