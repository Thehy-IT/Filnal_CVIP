import torch
import os

def collate_fn(batch):
    """
    Hàm này cực kỳ quan trọng cho Faster R-CNN.
    PyTorch mặc định cố gắng stack (xếp chồng) các tensor lại với nhau.
    Nhưng bounding boxes có độ dài khác nhau nên không stack được.
    Ta dùng hàm này để gộp chúng thành một Tuple.
    """
    return tuple(zip(*batch))

def save_checkpoint(state, filename="checkpoint.pth"):
    """Lưu model an toàn"""
    torch.save(state, filename)
    print(f"--> Đã lưu checkpoint tại {filename}")

def calculate_dice_score(preds, targets, smooth=1e-5):
    """
    Hàm tính Dice Score để đánh giá model Segmentation trên tập Validation.
    """
    # Đưa dự đoán qua sigmoid và làm tròn về nhị phân (0 hoặc 1) với ngưỡng 0.5
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    # Tính toán diện tích giao nhau (Intersection)
    intersection = (preds * targets).sum()
    
    # Công thức Dice: 2 * (A ∩ B) / (A + B)
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()