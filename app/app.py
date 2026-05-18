import streamlit as st
import os
import yaml
from PIL import Image
import sys

# Đảm bảo đường dẫn module gốc có thể import được
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import MedicalInferencePipeline

# Đọc cấu hình
def load_config(cfg_path="configs/config.yaml"):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

# --- CẤU HÌNH GIAO DIỆN (Tắt mọi icon, thiết lập layout rộng) ---
st.set_page_config(
    page_title="HỆ THỐNG PHÂN TÍCH TỔN THƯƠNG DA Y TẾ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM: Phong cách Clinical, Xanh Navy, Không Icon ---
st.markdown("""
    <style>
        /* Ẩn các menu mặc định của Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Tùy chỉnh màu nền và phông chữ */
        .reportview-container {
            background-color: #F4F6F9;
            color: #1E293B;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        /* Tiêu đề chính */
        .main-header {
            font-size: 28px;
            font-weight: 700;
            color: #0F172A;
            border-bottom: 2px solid #0284C7;
            padding-bottom: 10px;
            margin-bottom: 30px;
            text-transform: uppercase;
        }
        
        /* Tiêu đề phần tử */
        .sub-header {
            font-size: 18px;
            font-weight: 600;
            color: #334155;
            margin-bottom: 15px;
        }

        /* Tùy chỉnh nút bấm */
        .stButton>button {
            background-color: #0284C7;
            color: white;
            font-weight: bold;
            border-radius: 4px;
            border: none;
            padding: 10px 20px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0369A1;
            color: white;
        }
        
        /* Box kết quả */
        .result-box {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo Pipeline dự đoán (Cache lại để không load lại model mỗi lần ấn nút)
@st.cache_resource
def load_pipeline():
    config = load_config()
    # CHÚ Ý: Đường dẫn đến file weights bạn đã train. Nếu chưa train, hãy để model rỗng chạy thử UI.
    det_weight = config['detection']['weight_save_path']
    seg_weight = config['segmentation']['weight_save_path']
    image_size = config['detection'].get('image_size', 512)
    
    # Tạo thư mục và file giả nếu chưa có weights thực tế để UI không bị lỗi khi test
    os.makedirs(os.path.dirname(det_weight), exist_ok=True)
    os.makedirs(os.path.dirname(seg_weight), exist_ok=True)
    
    if not os.path.exists(det_weight) or not os.path.exists(seg_weight):
        st.warning("HỆ THỐNG: Chưa tìm thấy trọng số huấn luyện (.pth). Vui lòng huấn luyện mô hình trước. Giao diện đang chạy ở chế độ giả lập.")
        return None, image_size 
        
    return MedicalInferencePipeline(det_weight, seg_weight, image_size=image_size), image_size

pipeline, sys_image_size = load_pipeline()

# --- GIAO DIỆN CHÍNH ---
st.markdown('<div class="main-header">Hệ Thống Phân Tích Khối U Da (ISIC 2018)</div>', unsafe_allow_html=True)

# Chia layout 1 bên điều khiển, 1 bên hiển thị
col_control, col_display = st.columns([1, 3])

with col_control:
    st.markdown('<div class="sub-header">Bảng Điều Khiển</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Tải lên ảnh lâm sàng (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    confidence_thresh = st.slider("Ngưỡng nhận diện (Confidence)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    
    analyze_btn = st.button("TIẾN HÀNH PHÂN TÍCH")

with col_display:
    if uploaded_file is not None:
        # Hiển thị ảnh gốc nhanh
        original_image = Image.open(uploaded_file)
        
        if analyze_btn:
            if pipeline is None:
                st.error("LỖI KHỐI HỆ THỐNG: Không thể tiến hành suy luận do thiếu mô hình trọng số.")
            else:
                with st.spinner("Đang thực hiện phân tích bằng AI..."):
                    # Chạy suy luận
                    img_det, img_seg, raw_mask = pipeline.predict(original_image, det_threshold=confidence_thresh)
                    
                    st.markdown('<div class="sub-header">Báo Cáo Kết Quả Chuẩn Đoán</div>', unsafe_allow_html=True)
                    
                    # Chia 3 cột để hiển thị kết quả đồng thời
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.image(original_image.resize((sys_image_size, sys_image_size)), caption="Ảnh Gốc Lâm Sàng", use_column_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with res_col2:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.image(img_det, caption="Phát Hiện (Faster R-CNN)", use_column_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with res_col3:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.image(img_seg, caption="Phân Vùng (Attention U-Net)", use_column_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    # Hiển thị thông số kỹ thuật (Giả lập)
                    st.markdown("---")
                    st.markdown('<div class="sub-header">Thông Số Kỹ Thuật</div>', unsafe_allow_html=True)
                    stats_col1, stats_col2 = st.columns(2)
                    stats_col1.write("- Phương pháp phát hiện: Bounding Box (ResNet50-FPN)")
                    stats_col1.write(f"- Ngưỡng tin cậy áp dụng: {confidence_thresh}")
                    stats_col2.write("- Phương pháp phân vùng: Pixel-level Attention")
                    stats_col2.write(f"- Độ phân giải xử lý: {sys_image_size} x {sys_image_size} px")
    else:
        st.info("HỆ THỐNG ĐANG CHỜ DỮ LIỆU ĐẦU VÀO. Vui lòng tải ảnh lên ở bảng điều khiển bên trái.")