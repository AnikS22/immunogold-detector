"""Advanced loss functions for Nature-level particle detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class WeightedFocalLoss(nn.Module):
    """Focal loss with class weights for 6nm vs 12nm imbalance (11:1)."""
    def __init__(self, alpha=2.0, gamma=4.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights or {0: 1.0, 1: 11.0}
    
    def forward(self, pred, target):
        pred_prob = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        pos_loss = -target * torch.log(pred_prob) * (1 - pred_prob) ** self.alpha
        neg_loss = -(1 - target) * torch.log(1 - pred_prob) * pred_prob ** self.gamma
        weights = torch.where(target > 0.5, torch.full_like(target, self.class_weights.get(1, 1.0)), torch.full_like(target, self.class_weights.get(0, 1.0)))
        return (weights * (pos_loss + neg_loss)).mean()

class DiceLoss(nn.Module):
    """Boundary-aware Dice loss for precise particle edges."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return dice.mean()

class CenterNetAdvancedLoss(nn.Module):
    """Multi-task loss with class imbalance handling and boundary awareness."""

    def __init__(
        self,
        center_weight=1.0,
        class_weight=1.0,
        size_weight=0.1,
        offset_weight=1.0,
        conf_weight=1.0,
        boundary_weight=0.1,
        class_weights=None,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.center_weight = center_weight
        self.class_weight = class_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        self.conf_weight = conf_weight
        self.boundary_weight = boundary_weight
        self.class_weights = class_weights if class_weights is not None else {0: 1.0, 1: 11.0}
        self.label_smoothing = float(label_smoothing)
        self.weighted_focal = WeightedFocalLoss(class_weights=dict(self.class_weights))
        self.dice = DiceLoss()
        self.l1_loss = nn.L1Loss(reduction="none")
    
    @staticmethod
    def _focal_loss(pred, target, alpha=2.0, beta=4.0):
        # Wide clamp avoids log(0) / pow underflow in float16; loss should run in float32.
        pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
        one_m = 1.0 - pred
        pos_loss = -target * torch.log(pred) * (one_m**alpha)
        # Use pow with float32; pred**4 in FP16 often underflows and breaks training.
        neg_loss = -(1.0 - target) * torch.log(one_m) * torch.pow(pred, beta)
        return pos_loss + neg_loss

    def forward(self, predictions, targets):
        # Always compute in float32 (safe with AMP: forward may be FP16, loss is FP32).
        predictions = {k: v.float() for k, v in predictions.items()}
        target_centers = targets["centers"].float()
        loss = torch.zeros((), device=target_centers.device, dtype=torch.float32)

        pred_centers = torch.sigmoid(predictions["centers"])
        center_loss = self._focal_loss(pred_centers, target_centers).mean()
        loss = loss + self.center_weight * center_loss

        center_mask = target_centers > 0.5
        if center_mask.sum() > 0:
            pred_classes = predictions["classes"]
            target_classes = targets["class_ids"].long()
            w0, w1 = self.class_weights.get(0, 1.0), self.class_weights.get(1, 11.0)
            ce_weight = torch.tensor([w0, w1], device=pred_classes.device, dtype=torch.float32)
            class_loss = nn.functional.cross_entropy(
                pred_classes.float(),
                target_classes,
                weight=ce_weight,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            class_loss = (class_loss * center_mask.squeeze(1).float()).mean()
            loss = loss + self.class_weight * class_loss

            pred_sizes = predictions["sizes"].float()
            target_sizes = targets["sizes"].float()
            size_loss = self.l1_loss(pred_sizes, target_sizes)
            size_loss = (size_loss * center_mask.float()).mean()
            loss = loss + self.size_weight * size_loss

            pred_offsets = predictions["offsets"].float()
            target_offsets = targets["offsets"].float()
            offset_loss = self.l1_loss(pred_offsets, target_offsets)
            offset_loss = (offset_loss * center_mask.float()).mean()
            loss = loss + self.offset_weight * offset_loss

            pred_conf = torch.sigmoid(predictions["confidence"].float())
            target_conf = targets["confidence"].float()
            conf_loss = self.l1_loss(pred_conf, target_conf)
            conf_loss = (conf_loss * center_mask.float()).mean()
            loss = loss + self.conf_weight * conf_loss

        return loss
