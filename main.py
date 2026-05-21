import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.data_pipeline.dataset_det import ISICDetectionDataset
from src.data_pipeline.dataset_seg import ISICSegmentationDataset
from src.data_pipeline.transforms import get_det_transforms, get_seg_transforms
from src.engine.train_det import train_one_epoch_det, validate_det
from src.engine.train_seg import train_one_epoch_seg, validate_seg
from src.engine.utils import collate_fn, save_checkpoint
from src.models.unet import UNet
from src.models.faster_rcnn import get_detection_model
from src.models.loss import DiceBCELoss


def load_config(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def configure_torch(device):
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def seed_worker(worker_id):
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass


def get_num_workers(cfg):
    if "num_workers" in cfg:
        return int(cfg["num_workers"])
    cpu_count = os.cpu_count() or 2
    return max(0, min(4, cpu_count - 1))


def build_loader(dataset, cfg, shuffle, collate_fn_arg=None):
    num_workers = get_num_workers(cfg)
    kwargs = {
        "dataset": dataset,
        "batch_size": cfg["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("pin_memory", torch.cuda.is_available())),
        "collate_fn": collate_fn_arg,
    }

    if num_workers > 0:
        kwargs.update({
            "persistent_workers": bool(cfg.get("persistent_workers", True)),
            "prefetch_factor": int(cfg.get("prefetch_factor", 2)),
            "worker_init_fn": seed_worker,
        })

    return DataLoader(**kwargs)


def maybe_compile(model, cfg):
    if bool(cfg.get("compile", False)) and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def run_train_detection(cfg):
    print("=== TRAIN FASTER R-CNN (DETECTION) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch(device)
    use_amp = device.type == "cuda" and bool(cfg.get("amp", True))
    print(f"Device: {device} | AMP: {use_amp}")

    image_size = int(cfg.get("image_size", 512))
    train_dataset = ISICDetectionDataset(
        csv_file=cfg["train_csv"],
        img_dir=cfg["img_dir"],
        transforms=get_det_transforms(is_train=True, image_size=image_size),
    )
    val_dataset = ISICDetectionDataset(
        csv_file=cfg["val_csv"],
        img_dir=cfg["val_img_dir"],
        transforms=get_det_transforms(is_train=False, image_size=image_size),
    )

    train_loader = build_loader(train_dataset, cfg, shuffle=True, collate_fn_arg=collate_fn)
    val_loader = build_loader(val_dataset, cfg, shuffle=False, collate_fn_arg=collate_fn)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Workers: {get_num_workers(cfg)}")

    model = get_detection_model(
        num_classes=int(cfg.get("num_classes", 2)),
        detections_per_img=cfg.get("detections_per_img"),
    ).to(device)
    model = maybe_compile(model, cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        foreach=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(os.path.dirname(cfg["weight_save_path"]), exist_ok=True)
    best_map = 0.0
    validate_every = max(1, int(cfg.get("validate_every", 1)))

    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_one_epoch_det(model, optimizer, train_loader, device, epoch, scaler=scaler, use_amp=use_amp)
        print(f"Epoch [{epoch}/{cfg['epochs']}] - Train Loss: {loss:.4f}")

        if epoch % validate_every != 0 and epoch != cfg["epochs"]:
            continue

        mAP_50, mAP_50_95 = validate_det(model, val_loader, device, epoch, use_amp=use_amp)
        print(f"Validation mAP@0.5: {mAP_50:.4f} | mAP@0.5:0.95: {mAP_50_95:.4f}")

        if mAP_50 > best_map:
            best_map = mAP_50
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_map": best_map,
            }, cfg["weight_save_path"])
            print(f"Saved best checkpoint at epoch {epoch} with mAP@0.5: {best_map:.4f}")


def run_train_segmentation(cfg):
    print("=== TRAIN ATTENTION U-NET (SEGMENTATION) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch(device)
    use_amp = device.type == "cuda" and bool(cfg.get("amp", True))
    print(f"Device: {device} | AMP: {use_amp}")

    image_size = int(cfg.get("image_size", 512))
    train_dataset = ISICSegmentationDataset(
        img_dir=cfg["train_img_dir"],
        mask_dir=cfg["train_mask_dir"],
        transforms=get_seg_transforms(is_train=True, image_size=image_size),
    )
    val_dataset = ISICSegmentationDataset(
        img_dir=cfg["val_img_dir"],
        mask_dir=cfg["val_mask_dir"],
        transforms=get_seg_transforms(is_train=False, image_size=image_size),
    )

    train_loader = build_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Workers: {get_num_workers(cfg)}")

    model = UNet(img_ch=3, output_ch=1).to(device, memory_format=torch.channels_last)
    model = maybe_compile(model, cfg)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        foreach=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(os.path.dirname(cfg["weight_save_path"]), exist_ok=True)
    best_dice = 0.0
    validate_every = max(1, int(cfg.get("validate_every", 1)))

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch_seg(
            model, optimizer, criterion, train_loader, device, epoch, scaler=scaler, use_amp=use_amp
        )

        if epoch % validate_every != 0 and epoch != cfg["epochs"]:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
            continue

        val_loss, val_dice = validate_seg(model, criterion, val_loader, device, epoch, use_amp=use_amp)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model.state_dict(), cfg["weight_save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2018 training entrypoint")
    parser.add_argument("--task", type=str, required=True, choices=["train_det", "train_seg"])

    args = parser.parse_args()
    config = load_config()

    if args.task == "train_det":
        run_train_detection(config["detection"])
    elif args.task == "train_seg":
        run_train_segmentation(config["segmentation"])
