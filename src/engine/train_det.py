import math
import sys

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def train_one_epoch_det(model, optimizer, data_loader, device, epoch, scaler=None, use_amp: bool = False):
    model.train()
    total_loss = 0.0

    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Detection Train]", leave=True)

    for images, targets in loop:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"\nLoss is {loss_value}, stopping training")
            sys.exit(1)

        if scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        total_loss += loss_value
        loop.set_postfix(loss=loss_value)

    return total_loss / len(data_loader)


def validate_det(model, data_loader, device, epoch, use_amp: bool = False):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Detection Val]", leave=True)

    with torch.no_grad():
        for images, targets in loop:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(images)

            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds_cpu, targets_cpu)

    print("Dang tinh toan mAP...")
    mAP_dict = metric.compute()
    mAP_50 = mAP_dict["map_50"].item()
    mAP_50_95 = mAP_dict["map"].item()

    metric.reset()
    return mAP_50, mAP_50_95
