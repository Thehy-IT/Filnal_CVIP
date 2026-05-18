import os
import cv2
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, List, Dict

# Thiết lập logging chuyên nghiệp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình hằng số (Kích thước ảnh chuẩn đưa vào mô hình)
TARGET_SIZE = (512, 512)

def extract_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Trích xuất tọa độ Bounding Box từ Binary Mask.
    Trả về: (x_min, y_min, x_max, y_max) hoặc None nếu mask trống.
    """
    # Tìm các pixel có chứa tổn thương (> 0)
    coords = cv2.findNonZero(mask)
    
    if coords is None:
        return None
        
    # Lấy bounding box hình chữ nhật nhỏ nhất bao quanh tổn thương
    x, y, w, h = cv2.boundingRect(coords)
    
    # Trả về tọa độ tuyệt đối (x_min, y_min, x_max, y_max)
    return (x, y, x + w, y + h)

def process_split(
    split_name: str, 
    img_dir: Path, 
    mask_dir: Path, 
    out_img_dir: Path, 
    out_mask_dir: Path
) -> pd.DataFrame:
    """
    Xử lý một tập dữ liệu (Train hoặc Val).
    Bao gồm: Đọc, Resize, Lưu lại và Trích xuất Bbox.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy tất cả file .jpg trong thư mục input
    img_paths = list(img_dir.glob('*.jpg'))
    logger.info(f"Bắt đầu xử lý tập [{split_name}] - Tìm thấy {len(img_paths)} ảnh.")
    
    data_records: List[Dict] = []
    
    for img_path in tqdm(img_paths, desc=f"Processing {split_name}"):
        img_id = img_path.stem # Lấy tên file không có đuôi (VD: ISIC_0000000)
        mask_path = mask_dir / f"{img_id}_segmentation.png"
        
        if not mask_path.exists():
            logger.warning(f"Không tìm thấy mask cho ảnh {img_id}. Bỏ qua!")
            continue
            
        try:
            # 1. Đọc ảnh và mask
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                logger.error(f"Lỗi đọc file tại {img_id}. Bỏ qua!")
                continue
                
            # 2. Resize
            # Dùng INTER_LINEAR cho ảnh gốc để giữ độ mượt
            image_resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            # Dùng INTER_NEAREST cho mask để giữ nguyên giá trị nhị phân (0 hoặc 255)
            mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            
            # Đảm bảo mask hoàn toàn nhị phân sau khi resize (Đề phòng nhiễu)
            _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            
            # 3. Lấy Bounding Box từ mask ĐÃ resize
            bbox = extract_bbox_from_mask(mask_resized)
            
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                
                # 4. Lưu file đã xử lý ra thư mục mới
                cv2.imwrite(str(out_img_dir / f"{img_id}.jpg"), image_resized)
                cv2.imwrite(str(out_mask_dir / f"{img_id}.png"), mask_resized)
                
                # 5. Lưu thông tin tọa độ
                data_records.append({
                    'image_id': img_id,
                    'file_name': f"{img_id}.jpg",
                    'width': TARGET_SIZE[0],
                    'height': TARGET_SIZE[1],
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'class_id': 1 # Mặc định 1 class là Tổn thương (Lesion)
                })
        except Exception as e:
            logger.error(f"Lỗi không xác định khi xử lý {img_id}: {str(e)}")
            
    df = pd.DataFrame(data_records)
    return df

def main():
    # 1. Định nghĩa thư mục gốc dựa trên cấu trúc bạn đã cung cấp
    base_raw_dir = Path("data/raw")
    base_processed_dir = Path("data/processed")
    
    # 2. Cấu hình đường dẫn cho tập TRAINING
    train_img_dir = base_raw_dir / "ISIC2018_Task1-2_Training_Input"
    train_mask_dir = base_raw_dir / "ISIC2018_Task1_Training_GroundTruth"
    out_train_img = base_processed_dir / "train/images"
    out_train_mask = base_processed_dir / "train/masks"
    
    # 3. Cấu hình đường dẫn cho tập VALIDATION
    val_img_dir = base_raw_dir / "ISIC2018_Task1-2_Validation_Input"
    val_mask_dir = base_raw_dir / "ISIC2018_Task1_Validation_GroundTruth"
    out_val_img = base_processed_dir / "val/images"
    out_val_mask = base_processed_dir / "val/masks"

    logger.info("================ BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ================")
    
    # Xử lý tập Train
    train_df = process_split("Training", train_img_dir, train_mask_dir, out_train_img, out_train_mask)
    train_csv_path = base_processed_dir / "train_bboxes.csv"
    train_df.to_csv(train_csv_path, index=False)
    logger.info(f"Đã lưu nhãn Training tại: {train_csv_path} (Số lượng: {len(train_df)})")
    
    # Xử lý tập Validation
    val_df = process_split("Validation", val_img_dir, val_mask_dir, out_val_img, out_val_mask)
    val_csv_path = base_processed_dir / "val_bboxes.csv"
    val_df.to_csv(val_csv_path, index=False)
    logger.info(f"Đã lưu nhãn Validation tại: {val_csv_path} (Số lượng: {len(val_df)})")
    
    logger.info("================ HOÀN TẤT TIỀN XỬ LÝ ================")

if __name__ == "__main__":
    main()