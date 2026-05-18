import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.0):
        """
        inputs: Lô-git (Logits) dự đoán từ model (shape: B, 1, H, W)
        targets: Ground truth mask nhị phân (shape: B, 1, H, W)
        """
        # Áp dụng Sigmoid để đưa dự đoán về khoảng [0, 1]
        inputs = torch.sigmoid(inputs)       
        
        # Làm phẳng tensor (flatten)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 1. Tính toán Dice Loss
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        # 2. Tính toán BCE Loss (Dùng F.binary_cross_entropy thay vì BCEWithLogits 
        # vì ta đã qua Sigmoid ở bước trên)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Kết hợp tổng Loss
        Dice_BCE = bce_loss + dice_loss
        
        return Dice_BCE