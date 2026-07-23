import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config_loader import load_config

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(true_dist * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification."""

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def make_criterion(loss_cfg=None, class_weight=None):
    """Create a loss function from config."""
    cfg = load_config()
    
    if loss_cfg is None:
        loss_cfg = {}
        
    ltype = loss_cfg.get("type", cfg.training.loss_cfg.type).lower()

    if ltype == "ce":
        return nn.CrossEntropyLoss(weight=class_weight)

    if ltype == "lsce":
        return LabelSmoothingCrossEntropy(
            smoothing=loss_cfg.get("smoothing", cfg.training.loss_cfg.smoothing)
        )

    if ltype == "focal":
        return FocalLoss(
            gamma=loss_cfg.get("gamma", cfg.training.loss_cfg.focal_gamma),
            weight=class_weight,
        )

    raise ValueError(f"Unknown loss type: {ltype}")
