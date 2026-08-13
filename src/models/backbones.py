# src/models/backbones.py
from typing import Optional

import torch.nn as nn
from torchvision import models

from .torchvision_cache import build_torchvision_model


_KNOWN_BACKBONE_FEATURE_DIMS = {
    "resnet50": 2048,
    "swin_t": 768,
}


def known_feature_dim(backbone_name: str) -> int:
    feature_dim = _KNOWN_BACKBONE_FEATURE_DIMS.get(backbone_name)
    if feature_dim is None:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return feature_dim


def build_backbone(backbone_name: str, use_pretrained: bool) -> tuple[nn.Module, int, Optional[nn.Module]]:
    if backbone_name == "resnet50":
        res_weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        res = build_torchvision_model(models.resnet50, res_weights)
        backbone = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2,
            res.layer3,
            res.layer4,
        )
        return backbone, res.fc.in_features, None

    if backbone_name == "swin_t":
        swin_weights = models.Swin_T_Weights.IMAGENET1K_V1 if use_pretrained else None
        swin = build_torchvision_model(models.swin_t, swin_weights)
        return swin.features, swin.head.in_features, swin.norm

    raise ValueError(f"Unsupported backbone: {backbone_name}")


__all__ = ["build_backbone", "known_feature_dim"]
