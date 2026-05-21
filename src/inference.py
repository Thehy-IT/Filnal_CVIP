import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

# Import kiến trúc model
from src.models.faster_rcnn import get_detection_model
from src.models.unet import UNet

class MedicalInferencePipeline:
    def __init__(self, det_weight_path, seg_weight_path, image_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # 1. Tải mô hình Detection
        self.det_model = get_detection_model(num_classes=2)
        if torch.cuda.is_available():
            self.det_model.load_state_dict(torch.load(det_weight_path))
        else:
            self.det_model.load_state_dict(torch.load(det_weight_path, map_location='cpu'))
        self.det_model.to(self.device)
        self.det_model.eval()

        # 2. Tải mô hình Segmentation
        self.seg_model = UNet(img_ch=3, output_ch=1)
        if torch.cuda.is_available():
            self.seg_model.load_state_dict(torch.load(seg_weight_path))
        else:
            self.seg_model.load_state_dict(torch.load(seg_weight_path, map_location='cpu'))
        self.seg_model.to(self.device)
        self.seg_model.eval()

        # Tiền xử lý cho Segmentation (Cần Normalize giống lúc train)
        self.seg_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_pil, det_threshold=0.5):
        """
        Thực hiện chuỗi dự đoán: Nhận diện -> Phân vùng (Two-stage)
        """
        # Chuyển PIL Image sang OpenCV RGB
        img_np = np.array(image_pil.convert('RGB'))
        img_resized = cv2.resize(img_np, (self.image_size, self.image_size))
        
        # Tạo bản sao để vẽ kết quả
        img_bbox = img_resized.copy()
        img_overlay = img_resized.copy()
        
        # Khởi tạo mask rỗng toàn ảnh
        full_binary_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        with torch.no_grad():
            # --- 1. DETECTION PHASE ---
            # Faster R-CNN PyTorch nhận input là tensor scale [0-1]
            det_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(self.device)
            det_results = self.det_model(det_tensor)[0]

            boxes = det_results['boxes'].cpu().numpy()
            scores = det_results['scores'].cpu().numpy()

            # Lọc box có độ tin cậy cao nhất
            best_box = None
            if len(scores) > 0 and scores[0] >= det_threshold:
                best_box = boxes[0].astype(int)
                x_min, y_min, x_max, y_max = best_box
                
                # Cắt (clip) tọa độ để không bị văng ra khỏi ảnh
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(self.image_size, x_max), min(self.image_size, y_max)
                
                # Vẽ Bounding Box (Màu Đỏ y tế)
                cv2.rectangle(img_bbox, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(img_bbox, f"Tổn thương: {scores[0]:.2f}", (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # --- 2. SEGMENTATION PHASE (TWO-STAGE) ---
                # Chỉ segmentation trên vùng Bounding Box đã cắt (ROI)
                if x_max > x_min and y_max > y_min:
                    roi = img_resized[y_min:y_max, x_min:x_max]
                    roi_resized = cv2.resize(roi, (self.image_size, self.image_size))
                    
                    seg_tensor = self.seg_transform(roi_resized).unsqueeze(0).to(self.device)
                    seg_output = self.seg_model(seg_tensor)
                    
                    # Áp dụng Sigmoid và ngưỡng 0.5 để lấy Binary Mask cho ROI
                    prob_mask = torch.sigmoid(seg_output).squeeze().cpu().numpy()
                    roi_binary_mask = (prob_mask > 0.5).astype(np.uint8)
                    
                    # Resize mask của ROI về lại kích thước thực của Bounding Box trên ảnh gốc
                    roi_binary_mask_resized = cv2.resize(roi_binary_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                    
                    # Gắn mask của ROI vào mask tổng
                    full_binary_mask[y_min:y_max, x_min:x_max] = roi_binary_mask_resized

            # Tạo lớp Overlay màu Xanh lam (Cyan) lên vùng tổn thương
            colored_mask = np.zeros_like(img_resized)
            colored_mask[full_binary_mask == 1] = [0, 255, 255] # Màu Cyan
            
            # Trộn ảnh gốc và mask (Độ trong suốt 0.4)
            img_overlay = cv2.addWeighted(img_overlay, 1.0, colored_mask, 0.4, 0)
            
            # Vẽ thêm đường viền sắc nét bao quanh tổn thương
            contours, _ = cv2.findContours(full_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_overlay, contours, -1, (0, 255, 255), 1)

        return img_bbox, img_overlay, full_binary_mask