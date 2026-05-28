import streamlit as tf_st  # Tránh xung đột tên với tensorflow
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Vietnamese Food Identifier",
    page_icon="🍲",
    layout="centered"
)

# Định nghĩa danh sách class theo thứ tự chữ cái (khớp với ImageDataGenerator)
CLASS_NAMES = [
    'Banh mi', 
    'Banh trang nuong', 
    'Banh xeo', 
    'Bun bo hue', 
    'Com tam', 
    'Goi cuon', 
    'Pho'
]

# ==========================================
# CACHING MODEL (BẮT BUỘC CHO STREAMLIT CLOUD)
# ==========================================
@st.cache_resource
def load_my_model():
    """Load model một lần duy nhất và lưu vào bộ nhớ cache để tăng tốc độ app"""
    model_path = 'best_vietnamese_food_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Khởi chạy load model
with st.spinner("🔄 Đang tải mô hình AI... Vui lòng đợi trong giây lát..."):
    model = load_my_model()

# ==========================================
# HÀM XỬ LÝ VÀ DỰ ĐOÁN ẢNH
# ==========================================
def predict(image_data, model):
    # Định dạng lại kích thước ảnh chuẩn 224x224 như lúc train
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Chuyển ảnh thành mảng numpy và chuẩn hóa (rescale 1./255)
    img_array = np.asarray(image)
    
    # Kiểm tra nếu ảnh là ảnh trắng đen (1 channel) thì chuyển sang RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    # Nếu ảnh có kênh alpha (RGBA), chỉ lấy 3 kênh đầu (RGB)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = img_array / 255.0
    img_reshape = np.expand_dims(img_array, axis=0) # Thêm dimension cho batch size
    
    # Dự đoán
    prediction = model.predict(img_reshape)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx] * 100
    
    return CLASS_NAMES[class_idx], confidence

# ==========================================
# GIAO DIỆN ỨNG DỤNG (UI/UX)
# ==========================================
st.title("🍲 Nhận Diện Món Ăn Việt Nam")
st.write("Ứng dụng AI sử dụng mạng CNN để nhận diện 7 món ăn phổ biến.")
st.markdown("---")

# Tạo 2 tabs cho người dùng lựa chọn phương thức đầu vào
tab1, tab2 = st.tabs(["📸 Chụp ảnh từ Camera", "📤 Tải ảnh lên từ thiết bị"])

image = None

with tab1:
    camera_image = st.camera_input("Bấm nút chụp ảnh món ăn của bạn:")
    if camera_image:
        image = Image.open(camera_image)

with tab2:
    uploaded_file = st.file_uploader("Chọn một file ảnh (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# Xử lý khi có ảnh đầu vào
if image is not None:
    st.markdown("### 📷 Ảnh đầu vào của bạn:")
    st.image(image, use_container_width=True)
    
    st.markdown("---")
    with st.spinner("🧠 AI đang phân tích món ăn..."):
        label, confidence = predict(image, model)
    
    # Hiển thị kết quả một cách chuyên nghiệp
    st.subheader("📊 Kết quả dự đoán:")
    
    if confidence > 60: # Ngưỡng tin cậy chấp nhận được
        st.success(f"Dự đoán đây là món: **{label}**")
        st.info(f"🎯 Độ tự tin (Confidence): **{confidence:.2f}%**")
    else:
        st.warning(f"AI không quá chắc chắn, nhưng có khả năng cao là: **{label}** ({confidence:.2f}%)")
        st.caption("💡 Mẹo: Bạn hãy thử chụp lại ảnh ở góc độ rõ ràng và đủ ánh sáng hơn nhé!")