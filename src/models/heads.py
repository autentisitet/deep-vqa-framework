# src/models/heads.py
import torch.nn as nn


def build_quality_head(num_features: int, dropout_rate: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )


def init_head_weights(head: nn.Sequential) -> None:
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


__all__ = ["build_quality_head", "init_head_weights"]
