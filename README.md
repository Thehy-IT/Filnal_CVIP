# ISIC 2018 Skin Lesion Analysis (Phân Tích Khối U Da Y Tế)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Một hệ thống học máy (Computer Vision) kết hợp kỹ thuật **Phát hiện đối tượng (Object Detection)** và **Phân vùng ảnh (Image Segmentation)** nhằm hỗ trợ chẩn đoán và phân tích hình ảnh khối u trên da y tế, dựa trên bộ dữ liệu chuẩn **ISIC 2018**.

## Tính Năng Nổi Bật

- **Phát hiện đối tượng (Detection)**: Sử dụng kiến trúc `Faster R-CNN` với backbone ResNet50-FPN để định vị và khoanh vùng chính xác các tổn thương da thông qua Bounding box.
- **Phân vùng điểm ảnh (Segmentation)**: Ứng dụng mô hình `U-Net` (Attention U-Net) giúp tách nền và phân vùng chi tiết khu vực tổn thương da ở cấp độ pixel.
- **Data Pipeline Tối Ưu**: Tích hợp các module Augmentation tự động, tối ưu hóa quá trình tải dữ liệu bằng Dataloader, sử dụng AMP (Automatic Mixed Precision) tối ưu bộ nhớ vRAM GPU.
- **Giao diện Web Lâm Sàng**: Ứng dụng triển khai bằng `Streamlit` với giao diện chuyên nghiệp, hiện đại, cho phép bác sĩ/người dùng tải ảnh gốc lên và nhận kết quả trực quan ngay lập tức.

---

## Cấu Trúc Thư Mục Dự Án

```text
Filnal_CVIP/
├── app/                      # Giao diện người dùng Web App
│   └── app.py                # File chạy chính của giao diện Streamlit
├── configs/                  # Chứa file cấu hình trung tâm
│   └── config.yaml           # Cấu hình siêu tham số (Epochs, LR, Batch size, v.v)
├── data/                     # Thư mục chứa dữ liệu
│   ├── raw/                  # Dữ liệu gốc tải về (chưa qua xử lý)
│   └── processed/            # Dữ liệu sau khi làm sạch và chia tách
│       ├── train/            # Dữ liệu huấn luyện (images, masks)
│       ├── val/              # Dữ liệu kiểm thử (images, masks)
│       ├── train_bboxes.csv  # Nhãn tạo độ khung chuẩn đoán huấn luyện
│       └── val_bboxes.csv    # Nhãn tạo độ khung chuẩn đoán kiểm thử
├── experiments/              # Nơi lưu lại kết quả từ quá trình thực nghiệm
│   ├── logs/                 # Nhật ký hệ thống (Tensorboard, log text)
│   └── weights/              # Nơi lưu model checkpoints (file .pth)
├── notebooks/                # Các file Jupyter Notebook nghiên cứu thuật toán
│   ├── 01_EDA_and_Visualization.ipynb
│   └── 02_Test_Transforms.ipynb      
├── src/                      # Source Code chính của Pipeline
│   ├── data_pipeline/        # Xử lý luồng dữ liệu
│   │   ├── dataset_det.py    # Dataloader đặc thù cho Detection
│   │   ├── dataset_seg.py    # Dataloader đặc thù cho Segmentation
│   │   ├── preprocess.py     # Script tiền xử lý dữ liệu tự động
│   │   └── transforms.py     # Data augmentation logic
│   ├── engine/               # Chứa vòng lặp huấn luyện & đánh giá
│   │   ├── train_det.py      # Core training/val loop Detection
│   │   ├── train_seg.py      # Core training/val loop Segmentation
│   │   └── utils.py          # Hàm hỗ trợ lưu log, checkpoint
│   ├── models/               # Khởi tạo kiến trúc mạng nơ-ron
│   │   ├── faster_rcnn.py    # Cấu trúc mô hình Detection
│   │   ├── loss.py           # Các hàm Loss function (Dice, BCE)
│   │   └── unet.py           # Cấu trúc mô hình Segmentation
│   └── inference.py          # Script Inference dùng cho môi trường Web/Test
├── main.py                   # Điểm neo Entry-point (CLI) chạy toàn dự án
├── requirements.txt          # Các thư viện phụ thuộc
└── README.md                 # Tài liệu hướng dẫn sử dụng này
```

---

## Cài Đặt Môi Trường (Installation)

Yêu cầu hệ thống: **Python 3.8+** và phần cứng có hỗ trợ **CUDA** (Đề xuất có GPU Nvidia để huấn luyện mượt mà).

1. **Clone repository và di chuyển vào thư mục:**

   ```bash
   git clone <your-repo-url>
   cd Filnal_CVIP
   ```
2. **Khởi tạo môi trường ảo (Virtual Environment):**

   ```bash
   python -m venv venv
   # Kích hoạt trên Mac/Linux:
   source venv/bin/activate  
   # Kích hoạt trên Windows:
   venv\Scripts\activate   
   ```
3. **Cài đặt các gói phụ thuộc:**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Chuẩn Bị Dữ Liệu (Dataset)

1. Tải bộ dữ liệu gốc [ISIC 2018](https://challenge.isic-archive.com/data/) và giải nén vào thư mục `data/raw/`.
2. Chạy kịch bản phân loại, chia tập và tiền xử lý:
   ```bash
   python src/data_pipeline/preprocess.py
   ```

   *Script này sẽ tổ chức lại file và sinh ra `train_bboxes.csv`, `val_bboxes.csv` cùng các folder ảnh bên trong `data/processed/`.*

---

## Hướng Dẫn Sử Dụng (Usage)

### 1. Cấu Hình Hệ Thống

Mọi thông số (đường dẫn dữ liệu, Epochs, Batch size, LR, Threshold, Checkpoint path) đều quy định tại tệp trung tâm `configs/config.yaml`. Vui lòng thay đổi tại đây nếu bạn có nhu cầu Custom.

### 2. Huấn Luyện Mô Hình (Training)

Để thuận tiện, mọi luồng huấn luyện đều được điều phối thông qua CLI `main.py`.

- **Khởi chạy huấn luyện Phát Hiện Đối Tượng (Faster R-CNN):**
  ```bash
  python main.py --task train_det
  ```
- **Khởi chạy huấn luyện Phân Vùng Ảnh (U-Net):**
  ```bash
  python main.py --task train_seg
  ```

*Weights tốt nhất (highest mAP hoặc Dice score) sẽ được lưu tại `experiments/weights/`.*

### 3. Giao Diện Tương Tác Lâm Sàng (Web UI)

Sau khi có weights (hoặc tải weights pretrained bỏ vào mục `experiments/weights/`), chạy ứng dụng Web bằng Streamlit:

```bash
streamlit run app/app.py
```

- UI sẽ chạy tại `http://localhost:8501/`.
- Cung cấp tính năng tải ảnh trực tiếp để kiểm định mô hình nhận diện khung khối u (Bounding box) và mặt nạ (Segmentation Mask).

## Giấy Phép (License)

Dự án này được cấp phép theo tiêu chuẩn **MIT License**. Bạn được tự do sử dụng, sao chép, chỉnh sửa hoặc phân phối.

```text
MIT License

Copyright (c) 2024 Huỳnh Thế Hy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**Tác Giả:** Huỳnh Thế Hy
**MSSV:** 051205009083
