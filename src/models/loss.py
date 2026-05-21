import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.0):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

        probs = torch.sigmoid(inputs).contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        return bce_loss + dice_loss
