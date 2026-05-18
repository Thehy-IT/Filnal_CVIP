# Hướng Dẫn Chạy Toàn Bộ Dự Án (Execution Guide)

Dự án này là một hệ thống thị giác máy tính trong y tế (CVIP) với hai tác vụ chính là **Phát hiện (Detection)** bằng Faster R-CNN và **Phân vùng (Segmentation)** bằng Attention U-Net trên tập dữ liệu ISIC 2018.

Dưới đây là thứ tự các bước để chạy toàn bộ dự án từ lúc thiết lập môi trường cho đến khi khởi chạy giao diện web ứng dụng.

## 1. Thiết Lập Môi Trường (Environment Setup)

Trước tiên, bạn cần đảm bảo môi trường Python đã được thiết lập với các thư viện cần thiết.

```bash
# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 2. Tiền Xử Lý Dữ Liệu & Khám Phá (EDA - Tùy chọn)

Nếu dữ liệu nguyên bản chưa được chuẩn bị, bạn có thể thực hiện chạy các file Jupyter Notebook trong thư mục `notebooks/` để phân tích và chuẩn bị dữ liệu:

* Chạy `notebooks/01_EDA_and_Visualization.ipynb` để hiểu về tập dữ liệu, thống kê và xem xét dữ liệu đầu vào.
* Chạy `notebooks/02_Test_Transforms.ipynb` để kiểm tra các phương pháp tăng cường (augmentation) và tiền xử lý ảnh trước khi train.

*Lưu ý:* Mã nguồn giả định rằng sau bước này, dữ liệu huấn luyện và kiểm tra đã được xử lý và nằm tại thư mục `data/processed/` (gồm `train_bboxes.csv`, `val_bboxes.csv` và các thư mục `images`, `masks`). Các đường dẫn cụ thể được quy định tại `configs/config.yaml`.

## 3. Huấn Luyện Mô Hình (Model Training)

Dự án sử dụng `main.py` làm file thực thi chính cho quá trình huấn luyện. Các thông số như batch size, số lượng epochs, learning rate... có thể được điều chỉnh bên trong file `configs/config.yaml`.

Bạn sẽ cần chạy tuần tự hoặc độc lập hai tác vụ huấn luyện sau:

### 3.1. Huấn luyện mô hình Phát hiện (Detection - Faster R-CNN)

Mô hình này học cách vẽ bounding box quanh các vùng tổn thương.

```bash
python main.py --task train_det
```

* Trọng số tốt nhất sẽ được lưu tại: `experiments/weights/faster_rcnn_best.pth`

### 3.2. Huấn luyện mô hình Phân vùng (Segmentation - Attention U-Net)

Mô hình này học cách tạo mask chi tiết cấp độ điểm ảnh cho các vùng tổn thương.

```bash
python main.py --task train_seg
```

* Trọng số tốt nhất sẽ được lưu tại: `experiments/weights/attention_unet_best.pth`

## 4. Suy Luận & Khởi Chạy Web App (Inference & Deployment)

Sau khi đã có file trọng số (weights) của cả hai mô hình (hoặc ít nhất đã tạo ra file checkpoint sau khi train), bạn có thể tiến hành chạy ứng dụng web giao diện người dùng. Dự án sử dụng **Streamlit** để xây dựng UI.

Để khởi chạy giao diện:

```bash
streamlit run app/app.py
```

**Thao tác trên giao diện:**

1. Truy cập vào đường dẫn Local URL mà Streamlit cung cấp (thường là `http://localhost:8501`).
2. Tải lên một hình ảnh lâm sàng (định dạng JPG/PNG) qua Bảng Điều Khiển.
3. Chỉnh sửa Ngưỡng nhận diện (Confidence threshold) nếu cần.
4. Nhấn **"TIẾN HÀNH PHÂN TÍCH"** để hệ thống gọi mô hình thực hiện Object Detection và Segmentation, sau đó hiển thị kết quả trực quan trên màn hình.

---

**Tóm tắt luồng công việc (Workflow):**
`Cài đặt Requirements` ➔ `Chuẩn bị Data (Notebooks)` ➔ `Train Detection (main.py)` ➔ `Train Segmentation (main.py)` ➔ `Chạy Web App (app.py)`
