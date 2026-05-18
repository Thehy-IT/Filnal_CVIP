import cv2
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Any

class ISICDetectionDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transforms=None):
        """
        Args:
            csv_file: Đường dẫn đến file CSV chứa bounding boxes (VD: train_bboxes.csv)
            img_dir: Thư mục chứa ảnh đã xử lý
            transforms: Albumentations transforms
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        
        # Gom nhóm bounding boxes theo từng file ảnh
        self.image_groups = self.df.groupby('file_name')
        self.image_names = list(self.image_groups.groups.keys())

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_name = self.image_names[idx]
        group = self.image_groups.get_group(img_name)
        
        img_path = self.img_dir / img_name
        
        # Đọc ảnh (Albumentations yêu cầu format RGB)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lấy TẤT CẢ Bounding Boxes của ảnh
        boxes = group[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        labels = group['class_id'].values.tolist()

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        # Chuyển đổi sang Tensor chuẩn của PyTorch Detection
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.empty((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        return image, target
