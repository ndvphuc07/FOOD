import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Nhận diện món ăn Việt Nam", layout="centered")
st.title("📸 AI Nhận Diện Món Ăn Việt Nam")
st.write("Ứng dụng chạy trên nền tảng Streamlit Cloud")

# Tải model bằng TensorFlow kết hợp cache
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('best_vietnam_food_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Lỗi không thể tải file model .h5: {e}")

# Danh sách 5 món ăn xếp theo bảng chữ cái
labels = ['Banh mi', 'Banh trang nuong', 'Bun bo hue', 'Goi cuon', 'Pho']


# Cho phép cả chụp ảnh trực tiếp và upload ảnh từ máy
img_file = st.camera_input("Chụp ảnh món ăn:")
uploaded_file = st.file_uploader("Hoặc chọn một file ảnh từ máy:", type=["jpg", "jpeg", "png"])

# Xác định nguồn ảnh nào được chọn
target_file = img_file if img_file is not None else uploaded_file

if target_file is not None:
    image = Image.open(target_file)
    st.image(image, caption="Ảnh đầu vào", use_column_width=True)
    
    # Tiền xử lý ảnh đúng chuẩn lúc train
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img_array = np.asarray(image)
    
    # Ép kiểu và xử lý số kênh màu nếu là ảnh PNG trong suốt
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_reshape = img_array[np.newaxis, ...] / 255.0
    
    # Dự đoán
    with st.spinner("AI đang phân tích món ăn..."):
        prediction = model.predict(img_reshape)
        result_idx = np.argmax(prediction)
        score = prediction[0][result_idx] * 100
    
    # Hiển thị kết quả đẹp mắt
    st.success(f"Kết quả dự đoán: **{labels[result_idx]}**")
    st.metric(label="Độ tin cậy", value=f"{score:.2f}%")
