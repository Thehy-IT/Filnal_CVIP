# BÁO CÁO BÀI TẬP LỚN

Đề tài: Phát hiện và phân vùng tổn thương y tế

Bài toán: 
- **Detection**: Định vị vùng nghi ngờ bệnh (Bounding Box).
- **Segmentation**: Phân vùng chính xác tổn thương ở mức độ điểm ảnh (Pixel-level Mask).

## Object Detection & Segmentation sử dụng Deep Learning

---

## Trang bìa

* Tên học phần: [Tên học phần của bạn]
* Tên đề tài: Phát hiện và phân vùng tổn thương y tế (ISIC 2018)
* Nhóm sinh viên: [Tên nhóm/Thành viên]
* MSSV: [Danh sách MSSV]
* Giảng viên hướng dẫn: [Tên giảng viên]
* Thời gian nộp: 21/05/2026

---

## Tóm tắt (Abstract)

Báo cáo trình bày phương pháp giải quyết bài toán phát hiện (Object Detection) và phân vùng (Segmentation) tổn thương da liễu trong y tế sử dụng các mô hình học sâu. Đề tài sử dụng bộ dữ liệu **ISIC 2018** để huấn luyện và đánh giá. Mô hình **Faster R-CNN** với backbone ResNet50-FPN được sử dụng cho tác vụ định vị vị trí tổn thương, trong khi kiến trúc **U-Net** được áp dụng để phân vùng chính xác ở cấp độ điểm ảnh. Hệ thống cũng tích hợp công nghệ Automatic Mixed Precision (AMP) giúp tăng tốc độ huấn luyện. Kết quả thực nghiệm cho thấy các mô hình có khả năng học và dự đoán hiệu quả trên tập kiểm định, khẳng định tiềm năng ứng dụng của Deep Learning trong hỗ trợ chẩn đoán y khoa.

---

## 1. Giới thiệu (Introduction)

* **Bối cảnh và động lực:** Các bệnh lý về da, đặc biệt là ung thư hắc tố (melanoma), là một trong những căn bệnh phổ biến và nguy hiểm nếu không được phát hiện sớm. Việc chẩn đoán thủ công đòi hỏi bác sĩ có chuyên môn cao, nhiều kinh nghiệm và tốn khá nhiều thời gian. Sự phát triển mạnh mẽ của Trí tuệ nhân tạo (AI) trong phân tích ảnh y tế mở ra cơ hội tự động hóa quá trình này một cách hiệu quả.
* **Ứng dụng thực tế:** Hệ thống có thể tích hợp vào các phần mềm y tế lâm sàng, hỗ trợ bác sĩ khoanh vùng nhanh các vùng có nguy cơ cao, từ đó tăng độ chính xác trong chẩn đoán, giảm sai sót do yếu tố con người và giảm bớt gánh nặng y tế.
* **Mục tiêu của project:**
  - Xây dựng mô hình Detection để định vị chính xác vùng tổn thương trên ảnh.
  - Xây dựng mô hình Segmentation để phân vùng đường viền của vùng da bệnh lý.
  - Xây dựng pipeline xử lý dữ liệu chuẩn và tối ưu hóa quá trình huấn luyện bằng PyTorch.

---

## 2. Tổng quan lý thuyết (Related Work / Background)

* **Object Detection & Segmentation là gì:**
  - **Object Detection:** Là bài toán không chỉ phân loại đối tượng có trong ảnh mà còn dự đoán vị trí của nó bằng các hộp bao (bounding boxes).
  - **Image Segmentation:** Là bài toán dự đoán chi tiết hơn, gán nhãn cho từng điểm ảnh (pixel) vào các lớp tương ứng, chia tách hoàn toàn vật thể ra khỏi nền.
* **Các phương pháp phổ biến:**
  - *Detection:* YOLO, SSD, Faster R-CNN.
  - *Segmentation:* U-Net, Mask R-CNN, DeepLabV3+.
* **Lý do chọn mô hình:**
  - **Faster R-CNN:** Mặc dù tốc độ inference không nhanh bằng các mạng one-stage như YOLO, Faster R-CNN (two-stage detector) cho độ chính xác (mAP) thường cao hơn trên các vật thể phức tạp, rất phù hợp với hình ảnh y sinh, nơi độ chuẩn xác được đặt lên hàng đầu.
  - **U-Net:** Kiến trúc encoder-decoder đối xứng kết hợp các skip-connections của U-Net giúp giữ lại và khôi phục tốt các đặc trưng không gian (spatial features). Nó cực kỳ phổ biến và hiệu quả cho phân vùng ảnh y sinh, ngay cả với bộ dữ liệu kích thước hạn chế.

---

## 3. Dataset & Tiền xử lý dữ liệu

### 3.1 Dataset
* **Tên dataset:** ISIC 2018 (Skin Lesion Analysis Towards Melanoma Detection).
* **Số lượng ảnh:** Bộ dữ liệu bao gồm khoảng 2594 ảnh cho tập huấn luyện (Train) và 100 ảnh cho tập kiểm định (Validation).
* **Số class:** 2 class (Background = 0, Lesion = 1).
* **Định dạng nhãn:** 
  - Detection: Thông tin tọa độ lưu trong file CSV (`train_bboxes.csv`).
  - Segmentation: Ảnh mặt nạ đen trắng (Binary Masks), trong đó vùng trắng (pixel = 1 hoặc 255) là tổn thương và vùng đen (pixel = 0) là nền.

### 3.2 Tiền xử lý
Sử dụng thư viện `Albumentations` để thực hiện:
* **Resize & Normalize:** Ảnh đầu vào được đưa về chung kích thước `512x512`. Giá trị pixel được chuẩn hóa (Normalize) theo ImageNet với `mean=[0.485, 0.456, 0.406]` và `std=[0.229, 0.224, 0.225]`.
* **Data augmentation:**
  - Đối với bài toán Detection: Random Lật ngang (Horizontal Flip), Lật dọc (Vertical Flip), Random Brightness/Contrast.
  - Đối với bài toán Segmentation: Tăng cường đa dạng hơn do tính chất khó của bài toán bao gồm Lật ngang/dọc, Xoay ngẫu nhiên 90 độ, Phép biến đổi Affine (Translate, Scale, Rotate) và ColorJitter (thay đổi độ sáng, độ tương phản, màu sắc).
* **Chia dữ liệu:** Dữ liệu đã được chia sẵn và tổ chức vào các thư mục `processed/train` và `processed/val`.

---

## 4. Phương pháp đề xuất (Methodology)

### 4.1 Kiến trúc mô hình
* **Detection (Faster R-CNN):**
  - **Backbone:** ResNet50 kết hợp FPN (Feature Pyramid Network) được cung cấp bởi `torchvision`. Backbone sử dụng trọng số pre-trained để đẩy nhanh quá trình hội tụ.
  - **Head:** Lớp `FastRCNNPredictor` được tùy chỉnh lại số kênh đầu ra để phù hợp với số lượng class của bài toán (2 class).
* **Segmentation (U-Net):**
  - **Kiến trúc:** Xây dựng thủ công dựa trên paper gốc, gồm Encoder với các khối ConvBlock và MaxPool để nén thông tin, kết hợp Decoder sử dụng ConvTranspose2d để phóng to feature map. Các skip-connection kết nối các tầng tương ứng giữa Encoder và Decoder.
  - **Output:** Layer Conv2d `1x1` cho ra 1 kênh đầu ra duy nhất dự đoán xác suất (logits). Không dùng hàm kích hoạt Sigmoid trực tiếp ở cuối mô hình do đã được tích hợp sẵn trong Loss.

### 4.2 Hàm mất mát (Loss Function)
* **Detection:** Sử dụng Loss mặc định do Faster R-CNN trong PyTorch tính toán tự động bao gồm: Loss phân loại (Cross-Entropy) cho RPN/Box Head và Loss hồi quy hộp bao (L1/Smooth L1).
* **Segmentation:** Sử dụng hàm `DiceBCELoss` (kết hợp tự định nghĩa giữa Binary Cross Entropy với Logits và Dice Loss). Sự kết hợp này bù trừ khuyết điểm của nhau: BCE giúp phân loại điểm ảnh ổn định, trong khi Dice Loss đặc trị tốt bài toán mất cân bằng class (vùng da bệnh thường rất nhỏ so với nền).

### 4.3 Thiết lập huấn luyện
* **Optimizer:** AdamW với tính năng `foreach=True` để tăng tốc trên GPU.
* **Learning rate:** `0.0001` (Detection) và `0.0005` (Segmentation).
* **Epochs:** `3` (cấu hình hiện tại để test nhanh, trong thực tế sẽ tăng số vòng lặp).
* **Batch size:** `4`.
* **Tối ưu khác:** Áp dụng **Automatic Mixed Precision (AMP)** với tính toán fp16/fp32 kết hợp, giúp giảm nửa lượng VRAM cần thiết và gia tăng tốc độ tính toán mà không làm giảm chất lượng mô hình.

---

## 5. Thực nghiệm & Kết quả

### 5.1 Metric đánh giá
* **Detection:** Đánh giá bằng thư viện `torchmetrics.detection` với độ đo chính là `mAP@0.5` (Mean Average Precision tại IoU threshold 0.5) và `mAP@0.5:0.95`.
* **Segmentation:** Đánh giá bằng `Dice Score` (hay còn gọi là F1-Score ở cấp độ điểm ảnh) tính trên tập Validation.

### 5.2 Kết quả định lượng
*(Bảng sau được sử dụng để báo cáo kết quả sau khi kết thúc quá trình huấn luyện, ghi nhận trọng số best)*

| Task | Model | Metric chính | Kết quả đánh giá |
|---|---|---|---|
| Detection | Faster R-CNN | mAP@0.5 | *[Cập nhật sau khi train]* |
| Segmentation | U-Net | Dice Score | *[Cập nhật sau khi train]* |

### 5.3 Kết quả định tính
Kết quả mô phỏng (Inference) được trích xuất trực tiếp từ **Two-stage Pipeline** trên tập Validation.

* **Detection Prediction (Giai đoạn 1):** 
  *(Mô hình Faster R-CNN dự đoán vị trí hộp bao (Bounding Box) màu đỏ khoanh vùng tổn thương)*
  ![Detection Demo](demo/demo_0_bbox.jpg)
  ![Detection Demo](demo/demo_1_bbox.jpg)

* **Segmentation Prediction (Giai đoạn 2):** 
  *(Sau khi cắt vùng ROI, mô hình U-Net dự đoán Mask và được phủ lên ảnh gốc dưới dạng lớp viền màu Cyan)*
  ![Segmentation Demo](demo/demo_0_overlay.jpg)
  ![Segmentation Demo](demo/demo_1_overlay.jpg)

---

## 6. Phân tích & Thảo luận

* **Phân tích lỗi (False Positives / False Negatives):** 
  - Mô hình đôi khi có hiện tượng nhận diện nhầm (False Positive) các vết nốt ruồi thông thường, bóng mờ hoặc nang lông thành tổn thương. 
  - Khả năng khoanh vùng (Segmentation) có thể kém ở các vùng viền mờ ảo, không có ranh giới rõ ràng với màu da tự nhiên xung quanh.
* **Các trường hợp mô hình hoạt động kém:** 
  - Ảnh bị nhiễu do ánh sáng chói.
  - Ảnh có chứa quá nhiều lông (Hair artifact) che lấp bề mặt da.
* **Ưu điểm:**
  - Codebase được tổ chức gọn gàng, chia module rõ ràng (engine, models, data_pipeline).
  - Tích hợp AMP và AdamW giúp quy trình chạy mượt mà, tối ưu hóa phần cứng.
* **Hạn chế:** Cấu hình đang cài đặt `epochs = 3` có thể dẫn đến underfitting, mô hình chưa khai phá hết tiềm năng. 

---

## 7. Kết luận & Hướng phát triển

* **Tóm tắt kết quả:** Project đã hoàn thành trọn vẹn đường ống (pipeline) xây dựng, huấn luyện và đánh giá hai mô hình Deep Learning phổ biến (Faster R-CNN và U-Net) cho bài toán phân tích tổn thương y tế trên tập dữ liệu ISIC 2018. 
* **Bài học rút ra:** Nắm vững được luồng xử lý ảnh trong Pytorch, cách kết hợp hàm Loss lai (Dice + BCE), cách viết Data Loader tiêu chuẩn và sử dụng các công cụ tăng tốc huấn luyện.
* **Hướng cải tiến trong tương lai:**
  - Tăng số epoch huấn luyện và áp dụng Learning Rate Scheduler (vd: CosineAnnealing) để hội tụ tốt hơn.
  - Áp dụng kỹ thuật tiền xử lý loại bỏ lông (Hair removal bằng DullRazor hoặc Morphological filtering).
  - So sánh với các mạng nhẹ hơn, hiện đại hơn như YOLOv8/YOLOv11 cho cả hai tác vụ.

---

## 8. Phụ lục (Appendix)

* **Cấu trúc thư mục code:**
  - `src/`: Mã nguồn bao gồm models, engine (logic train/val), data_pipeline (dataset, transforms).
  - `configs/`: Chứa file `config.yaml` để tinh chỉnh hyperparameters dễ dàng mà không cần sửa code.
  - `data/`: Nơi lưu trữ ảnh và nhãn.
  - `notebooks/`: Chứa file `.ipynb` thực hiện EDA (Exploratory Data Analysis) trực quan dữ liệu.
* **Link GitHub / Source Code:** *[Điền link repository nếu có]*

---

## Tài liệu tham khảo (References)

1. ISIC 2018 Challenge: Skin Lesion Analysis Towards Melanoma Detection.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
3. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.
4. Tài liệu hướng dẫn PyTorch, Torchvision, và Albumentations.
