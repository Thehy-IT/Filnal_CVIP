import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader

# Import từ các module ta đã viết
from src.data_pipeline.transforms import get_det_transforms, get_seg_transforms
from src.data_pipeline.dataset_det import ISICDetectionDataset
from src.data_pipeline.dataset_seg import ISICSegmentationDataset

from src.models.faster_rcnn import get_detection_model
from src.models.attention_unet import AttentionUNet
from src.models.loss import DiceBCELoss

from src.engine.utils import collate_fn, save_checkpoint
from src.engine.train_det import train_one_epoch_det, validate_det
from src.engine.train_seg import train_one_epoch_seg, validate_seg

def load_config(cfg_path="configs/config.yaml"):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def run_train_detection(cfg):
    print("=== KHỞI ĐỘNG TRAINING FASTER R-CNN (DETECTION) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # 1. Chuẩn bị Dữ liệu
    train_dataset = ISICDetectionDataset(
        csv_file=cfg['train_csv'], 
        img_dir=cfg['img_dir'], 
        transforms=get_det_transforms(is_train=True)
    )
    val_dataset = ISICDetectionDataset(
        csv_file=cfg['val_csv'], 
        img_dir=cfg['val_img_dir'], 
        transforms=get_det_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn, # Bắt buộc đối với Faster R-CNN
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )

    # 2. Khởi tạo Model & Optimizer
    model = get_detection_model(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

    # 3. Vòng lặp Huấn luyện
    os.makedirs(os.path.dirname(cfg['weight_save_path']), exist_ok=True)
    best_map = 0.0

    for epoch in range(1, cfg['epochs'] + 1):
        loss = train_one_epoch_det(model, optimizer, train_loader, device, epoch)
        print(f"Epoch [{epoch}/{cfg['epochs']}] - Train Loss: {loss:.4f}")
        
        # Đánh giá trên tập validation
        mAP_50, mAP_50_95 = validate_det(model, val_loader, device, epoch)
        print(f"Validation mAP@0.5: {mAP_50:.4f} | mAP@0.5:0.95: {mAP_50_95:.4f}")

        # Lưu model nếu mAP@0.5 tăng
        if mAP_50 > best_map:
            best_map = mAP_50
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map
            }, cfg['weight_save_path'])
            print(f"Đã lưu checkpoint tốt nhất tại epoch {epoch} với mAP@0.5: {best_map:.4f}")

def run_train_segmentation(cfg):
    print("=== KHỞI ĐỘNG TRAINING ATTENTION U-NET (SEGMENTATION) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # 1. Chuẩn bị Dữ liệu
    train_dataset = ISICSegmentationDataset(
        img_dir=cfg['train_img_dir'], mask_dir=cfg['train_mask_dir'], transforms=get_seg_transforms(is_train=True, image_size=cfg.get('image_size', 512))
    )
    val_dataset = ISICSegmentationDataset(
        img_dir=cfg['val_img_dir'], mask_dir=cfg['val_mask_dir'], transforms=get_seg_transforms(is_train=False, image_size=cfg.get('image_size', 512))
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # 2. Khởi tạo Model, Loss & Optimizer
    model = AttentionUNet(img_ch=3, output_ch=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])

    # 3. Vòng lặp Huấn luyện
    os.makedirs(os.path.dirname(cfg['weight_save_path']), exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_one_epoch_seg(model, optimizer, criterion, train_loader, device, epoch)
        val_loss, val_dice = validate_seg(model, criterion, val_loader, device, epoch)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Lưu model có Dice Score trên tập Validate cao nhất
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model.state_dict(), cfg['weight_save_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chương trình huấn luyện ISIC 2018")
    parser.add_argument('--task', type=str, required=True, choices=['train_det', 'train_seg'], 
                        help="Chọn 'train_det' (Phát hiện) hoặc 'train_seg' (Phân vùng)")
    
    args = parser.parse_args()
    config = load_config()

    if args.task == 'train_det':
        run_train_detection(config['detection'])
    elif args.task == 'train_seg':
        run_train_segmentation(config['segmentation'])