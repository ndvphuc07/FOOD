"""
Vietnamese Food Recognition Web Application
Author: Senior AI & Software Engineer Persona
Framework: Streamlit & TensorFlow (Optimized for Streamlit Cloud Deployment)
"""

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# ==============================================================================
# 1. APPLICATION & UI CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AI Vietnamese Food Recognizer",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS để giao diện scannable và chuyên nghiệp hơn
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; color: #005088; text-align: center; font-weight: 700; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #555555; margin-bottom: 30px; }
    .result-box { padding: 20px; border-radius: 10px; background-color: #f0f8ff; border-left: 5px solid #11CAA0; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📸 Hệ Thống Nhận Diện Món Ăn Việt Nam</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ứng dụng Deep Learning (CNN) nhận diện Phở, Bánh mì, Bún bò Huế, Gỏi cuốn, Bánh tráng nướng</div>', unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL INFRASTRUCTURE & CACHING
# ==============================================================================
MODEL_PATH = 'best_vietnam_food_model.h5'
TARGET_IMAGE_SIZE = (224, 224)
# Thứ tự nhãn được chuẩn hóa nghiêm ngặt theo cấu trúc phân lớp từ bộ dữ liệu gốc
FOOD_LABELS = ['Banh mi', 'Banh trang nuong', 'Bun bo hue', 'Goi cuon', 'Pho']

@st.cache_resource(show_spinner=False)
def initialize_weights(model_file: str) -> tf.keras.Model:
    """Tải và cấu hình mô hình CNN vào bộ nhớ đệm (Cache) của hệ thống."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Không tìm thấy file trọng số cấu hình '{model_file}' tại thư mục gốc.")
    return tf.keras.models.load_model(model_file)

# Khởi tạo mô hình an toàn với khối try-except tuần tự
try:
    with st.spinner("Đang khởi tạo cấu trúc mạng CNN và nạp trọng số mô hình..."):
        model = initialize_weights(MODEL_PATH)
except Exception as error:
    st.error(f"❌ [System Error] Không thể tải cấu trúc mô hình. Chi tiết lỗi hệ thống: {error}")
    st.stop()

# ==============================================================================
# 3. PIPELINE TIỀN XỬ LÝ ẢNH (IMAGE PREPROCESSING)
# ==============================================================================
def preprocess_input_image(raw_image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Chuẩn hóa ảnh đầu vào trước khi nạp vào mạng Neural Network.
    - Ép định dạng màu sang RGB (xử lý triệt để lỗi kênh màu Alpha trên ảnh PNG).
    - Smart Resize ảnh giữ nguyên tỷ lệ không méo hình (Lanczos interpolation).
    - Chuẩn hóa phân đoạn dải điểm ảnh (Rescale về khoảng [0, 1]).
    - Ép mảng 3D thành Tensor 4D (Batch size = 1).
    """
    # Bước 1: Khử kênh alpha nếu có
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert("RGB")
        
    # Bước 2: Thay đổi kích thước thông minh theo chuẩn input đầu vào của mô hình
    processed_img = ImageOps.fit(raw_image, target_size, Image.LANCZOS)
    
    # Bước 3: Chuyển dữ liệu sang dạng ma trận mảng số thực và co giãn phân rã ma trận
    img_array = np.asarray(processed_img, dtype=np.float32)
    rescaled_img = img_array / 255.0
    
    # Bước 4: Mở rộng chiều ma trận (Thêm trục Batch Dimension)
    final_tensor = np.expand_dims(rescaled_img, axis=0)
    return final_tensor

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG & LUỒNG THỰC THI (UX & COMPONENT LOGIC)
# ==============================================================================
# Cung cấp 2 cơ chế tương tác đa dạng hóa nguồn tài nguyên ảnh đầu vào
mode_selection = st.radio("Chọn phương thức nhập dữ liệu ảnh:", ("Chụp ảnh trực tiếp", "Tải file ảnh lên từ máy"))

captured_buffer = None
uploaded_buffer = None

if mode_selection == "Chụp ảnh trực tiếp":
    captured_buffer = st.camera_input("Vui lòng căn chỉnh món ăn giữa khung hình camera:")
else:
    uploaded_buffer = st.file_uploader("Chọn tệp tin hình ảnh món ăn:", type=["jpg", "jpeg", "png"])

# Hợp nhất nguồn dữ liệu đệm từ camera hoặc tệp tải lên
active_image_buffer = captured_buffer if captured_buffer is not None else uploaded_buffer

if active_image_buffer is not None:
    try:
        # Đọc luồng dữ liệu nhị phân thành đối tượng Image
        user_image = Image.open(active_image_buffer)
        
        # Tạo khu vực hiển thị ảnh xem trước sắc nét cho người dùng
        st.markdown("---")
        st.subheader("🖼️ Dữ liệu hình ảnh đầu vào")
        st.image(user_image, caption="Hình ảnh thực tế được cung cấp", use_column_width=True)
        
        # Kích hoạt tiến trình phân tích xử lý dữ liệu ảnh
        with st.spinner("Hệ thống đang trích xuất đặc trưng hình ảnh và tính toán phân lớp..."):
            # Chạy pipeline tiền xử lý hình ảnh
            input_tensor = preprocess_input_image(user_image, TARGET_IMAGE_SIZE)
            
            # Thực thi suy luận (Inference) từ mạng CNN
            raw_predictions = model.predict(input_tensor)
            
            # Trích xuất chỉ số phân lớp có xác suất phân phối lớn nhất (Argmax)
            top_class_index = np.argmax(raw_predictions)
            confidence_score = raw_predictions[0][top_class_index] * 100
            
        # ==============================================================================
        # 5. KẾT QUẢ ĐẦU RA (METRICS & LOGS REPORT)
        # ==============================================================================
        st.subheader("🎯 Kết quả phân tích từ AI")
        
        # Ngưỡng tin cậy hệ thống (Confidence Threshold = 40%) để chặn ảnh rác/ảnh không liên quan
        if confidence_score > 40.0:
            predicted_label = FOOD_LABELS[top_class_index]
            
            # Hiển thị giao diện kết quả chuẩn scannable trực quan
            st.markdown(f"""
                <div class="result-box">
                    <h3 style='margin: 0; color: #005088;'>Món ăn được nhận diện: <b>{predicted_label}</b></h3>
                    <p style='margin: 10px 0 0 0; color: #333;'>Hệ thống đã khớp cấu trúc hình ảnh chính xác.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Thanh hiển thị định lượng phần trăm độ tin cậy trực quan
            st.write("")
            st.metric(label="Độ chính xác / Tin cậy của mô hình", value=f"{confidence_score:.2f}%")
            st.progress(int(confidence_score))
        else:
            st.warning("⚠️ **Hệ thống từ chối phân lớp:** Độ tin cậy quá thấp hoặc hình ảnh cung cấp không nằm trong danh mục 5 món ăn hệ thống được huấn luyện. Vui lòng chụp lại ảnh rõ nét hơn!")
            
    except Exception as run_error:
        st.error(f"❌ Đã xảy ra sự cố trong quá trình tiền xử lý ảnh hoặc thực thi dự đoán: {run_error}")