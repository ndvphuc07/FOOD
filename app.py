"""
Vietnamese Food Recognition Web Application
Author: Professional Software Engineer Persona
Feature: Integrated Dynamic H5 Hot-Patching for Keras 2/3 Cross-Compatibility
"""

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import h5py
import json

# ==============================================================================
# 1. APPLICATION & UI CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AI Vietnamese Food Recognizer",
    page_icon="📸",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; color: #005088; text-align: center; font-weight: 700; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #555555; margin-bottom: 30px; }
    .result-box { padding: 20px; border-radius: 10px; background-color: #f0f8ff; border-left: 5px solid #11CAA0; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📸 Hệ Thống Nhận Diện Món Ăn Việt Nam</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ứng dụng Deep Learning (CNN) nhận diện Phở, Bánh mì, Bún bò Huế, Gỏi cuốn, Bánh tráng nướng</div>', unsafe_allow_html=True)

MODEL_PATH = 'best_vietnam_food_model.h5'
TARGET_IMAGE_SIZE = (224, 224)
FOOD_LABELS = ['Banh mi', 'Banh trang nuong', 'Bun bo hue', 'Goi cuon', 'Pho']

# ==============================================================================
# 2. ELITE UTILITY: Keras 3 to Keras 2 Hot-Patching
# ==============================================================================
def hot_patch_keras_model(h5_path: str):
    """
    Trích xuất chuỗi JSON cấu hình kiến trúc mạng bên trong file .h5,
    đệ quy loại bỏ toàn bộ các tham số độc quyền của Keras 3 để bảo toàn tính 
    tương thích ngược trên môi trường Keras 2 của Streamlit Cloud.
    """
    if not os.path.exists(h5_path):
        return
    try:
        # Mở file .h5 ở chế độ đọc-ghi (Read/Write)
        with h5py.File(h5_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_data = f.attrs['model_config']
                if isinstance(config_data, bytes):
                    config_data = config_data.decode('utf-8')
                
                config_json = json.loads(config_data)
                
                # Hàm đệ quy duyệt cây cấu trúc JSON để xóa các key không hợp lệ
                def clean_unsupported_nodes(node):
                    if isinstance(node, dict):
                        # Danh sách các tham số gây lỗi bất đồng bộ thiết kế lớp giữa 2 phiên bản
                        bad_keys = ['quantization_config', 'optional', 'batch_shape', 'ragged', 'sparse']
                        for key in bad_keys:
                            node.pop(key, None)
                        for k, v in node.items():
                            clean_unsupported_nodes(v)
                    elif isinstance(node, list):
                        for item in node:
                            clean_unsupported_nodes(item)
                
                clean_unsupported_nodes(config_json)
                # Ghi đè lại chuỗi cấu hình sạch vào thuộc tính metadata của file h5
                f.attrs['model_config'] = json.dumps(config_json).encode('utf-8')
    except Exception as patch_err:
        # Nếu file đang bị khóa hoặc đã được patch trước đó, bỏ qua để tránh crash luồng
        pass

# ==============================================================================
# 3. MODEL LOADING WITH CACHE
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_and_initialize_model(model_file: str) -> tf.keras.Model:
    # Thực hiện vá lỗi tệp tin cấu hình trước khi nạp vào bộ nhớ
    hot_patch_keras_model(model_file)
    return tf.keras.models.load_model(model_file)

try:
    with st.spinner("Hệ thống đang thực hiện Vá lỗi cấu trúc cấu hình và Nạp trọng số mô hình..."):
        model = load_and_initialize_model(MODEL_PATH)
except Exception as error:
    st.error(f"❌ [System Error] Không thể giải tuần tự hóa mô hình: {error}")
    st.stop()

# ==============================================================================
# 4. PIPELINE TIỀN XỬ LÝ ẢNH (IMAGE PREPROCESSING)
# ==============================================================================
def preprocess_input_image(raw_image: Image.Image, target_size: tuple) -> np.ndarray:
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert("RGB")
    processed_img = ImageOps.fit(raw_image, target_size, Image.LANCZOS)
    img_array = np.asarray(processed_img, dtype=np.float32)
    rescaled_img = img_array / 255.0
    final_tensor = np.expand_dims(rescaled_img, axis=0)
    return final_tensor

# ==============================================================================
# 5. UX COMPONENTS & INFERENCE FLOW
# ==============================================================================
mode_selection = st.radio("Chọn phương thức nhập dữ liệu ảnh:", ("Chụp ảnh trực tiếp", "Tải file ảnh lên từ máy"))
active_image_buffer = None

if mode_selection == "Chụp ảnh trực tiếp":
    active_image_buffer = st.camera_input("Vui lòng căn chỉnh món ăn giữa khung hình camera:")
else:
    active_image_buffer = st.file_uploader("Chọn tệp tin hình ảnh món ăn:", type=["jpg", "jpeg", "png"])

if active_image_buffer is not None:
    try:
        user_image = Image.open(active_image_buffer)
        st.markdown("---")
        st.subheader("🖼️ Dữ liệu hình ảnh đầu vào")
        st.image(user_image, caption="Hình ảnh thực tế được cung cấp", use_column_width=True)
        
        with st.spinner("AI đang trích xuất đặc trưng hình ảnh..."):
            input_tensor = preprocess_input_image(user_image, TARGET_IMAGE_SIZE)
            raw_predictions = model.predict(input_tensor)
            top_class_index = np.argmax(raw_predictions)
            confidence_score = raw_predictions[0][top_class_index] * 100
            
        st.subheader("🎯 Kết quả phân tích từ AI")
        
        # Ngưỡng lọc chặn ảnh rác ngoài danh mục (40%)
        if confidence_score > 40.0:
            predicted_label = FOOD_LABELS[top_class_index]
            st.markdown(f"""
                <div class="result-box">
                    <h3 style='margin: 0; color: #005088;'>Món ăn được nhận diện: <b>{predicted_label}</b></h3>
                    <p style='margin: 10px 0 0 0; color: #333;'>Hệ thống đã khớp cấu trúc hình ảnh chính xác.</p>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.metric(label="Độ chính xác / Tin cậy của mô hình", value=f"{confidence_score:.2f}%")
            st.progress(int(confidence_score))
        else:
            st.warning("⚠️ **Hệ thống từ chối phân lớp:** Hình ảnh không rõ ràng hoặc không nằm trong danh mục 5 món ăn hệ thống được học.")
            
    except Exception as run_error:
        st.error(f"❌ Lỗi thực thi dự đoán hệ thống: {run_error}")
