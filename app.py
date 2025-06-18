# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# List kelas sesuai dataset kamu
CLASS_NAMES = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']
IMG_SIZE = (150, 150)
MODEL_PATH = "corn_leaf_disease_model.h5"

# Fungsi untuk load model
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di {MODEL_PATH}")
        return None
    return load_model(MODEL_PATH)

# Fungsi prediksi gambar
def predict_image(img, model):
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    pred_label = CLASS_NAMES[pred_index]
    confidence = float(prediction[pred_index])

    return pred_label, confidence

# Tampilan Streamlit
st.title("🌽 Deteksi Penyakit Daun Jagung")
st.write("Upload gambar daun jagung, lalu sistem akan mendeteksi penyakit berdasarkan model yang sudah dilatih.")

# Load model
model = load_cnn_model()

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload Gambar Daun Jagung", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Gambar yang Diupload", use_column_width=True)

    if st.button("🔍 Prediksi Gambar"):
        with st.spinner("Sedang memproses..."):
            label, confidence = predict_image(image, model)
            st.success(f"✅ Hasil Prediksi: **{label}** dengan keyakinan **{confidence * 100:.2f}%**")
