import torch
import math
import sys
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train_one_epoch_det(model, optimizer, data_loader, device, epoch):
    """
    Huấn luyện Faster R-CNN trong 1 Epoch
    """
    model.train() # Chuyển model sang chế độ train (Rất quan trọng)
    total_loss = 0
    
    # Thanh tiến trình chuyên nghiệp
    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Detection Train]", leave=True)
    
    for images, targets in loop:
        # Đưa dữ liệu lên GPU/CPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 1. Truyền tiến (Forward pass)
        # Faster R-CNN tự động tính loss khi ở chế độ model.train()
        loss_dict = model(images, targets)
        
        # Tính tổng các loại loss (classifier, box_reg, objectness, rpn_box_reg)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Kiểm tra nếu loss bị NaN hoặc Infinity thì dừng ngay lập tức
        if not math.isfinite(loss_value):
            print(f"\nLoss is {loss_value}, stopping training")
            sys.exit(1)
            
        # 2. Truyền ngược (Backward pass)
        optimizer.zero_grad() # Xóa gradient cũ
        losses.backward()     # Tính gradient mới
        optimizer.step()      # Cập nhật trọng số (Weights)
        
        total_loss += loss_value
        
        # Cập nhật thanh tiến trình
        loop.set_postfix(loss=loss_value)
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def validate_det(model, data_loader, device, epoch):
    """
    Đánh giá Faster R-CNN trên tập Validation bằng mAP
    """
    model.eval() # Chuyển sang chế độ đánh giá
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Detection Val]", leave=True)
    
    with torch.no_grad():
        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Predict
            preds = model(images)
            
            # Move predictions and targets to CPU to calculate metrics efficiently
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            metric.update(preds_cpu, targets_cpu)
            
    # Calculate mAP
    print("Đang tính toán mAP (có thể mất thời gian)...")
    mAP_dict = metric.compute()
    mAP_50 = mAP_dict['map_50'].item()
    mAP_50_95 = mAP_dict['map'].item()
    
    metric.reset() # Reset for next epoch
    
    return mAP_50, mAP_50_95
