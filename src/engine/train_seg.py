import torch
from tqdm import tqdm
from src.engine.utils import calculate_dice_score

def train_one_epoch_seg(model, optimizer, criterion, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Seg Train]", leave=True)
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        preds = model(images)
        loss = criterion(preds, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(data_loader)

def validate_seg(model, criterion, data_loader, device, epoch):
    model.eval() # Chuyển sang chế độ đánh giá (Tắt Dropout, BatchNorm tĩnh)
    total_loss = 0
    total_dice = 0
    
    # Tắt tính toán Gradient để tiết kiệm bộ nhớ và tăng tốc
    with torch.no_grad():
        loop = tqdm(data_loader, desc=f"Epoch {epoch} [Seg Val]", leave=True)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            preds = model(images)
            loss = criterion(preds, masks)
            dice = calculate_dice_score(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice
            
            loop.set_postfix(loss=loss.item(), dice=dice)
            
    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / len(data_loader)
    
    return avg_loss, avg_dice