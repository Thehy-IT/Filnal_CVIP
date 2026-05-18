import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset

class ISICSegmentationDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, transforms=None):
        """
        Args:
            img_dir: Thư mục chứa ảnh
            mask_dir: Thư mục chứa mask
            transforms: Albumentations transforms
        """
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        
        # Lấy danh sách tên file hợp lệ (chỉ lấy những ảnh có tồn tại mask)
        self.images = [
            f.name for f in self.img_dir.glob('*.jpg') 
            if (self.mask_dir / f.name.replace('.jpg', '.png')).exists()
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        img_path = self.img_dir / img_name
        mask_path = self.mask_dir / mask_name
        
        # Đọc ảnh (RGB) và Mask (Grayscale)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Đưa mask về dạng nhị phân chuẩn 0.0 và 1.0
        mask = (mask > 127).astype("float32")

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
            # Albumentations trả về mask shape là (H, W), U-Net cần (1, H, W)
            mask = mask.unsqueeze(0)

        return image, mask