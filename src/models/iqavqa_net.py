# src/models/iqavqa_net.py
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src.config.schemas import Config


class IQAVQANet(nn.Module):
    """
    Quality assessment network supporting both images and videos.

    - Image (4D): backbone -> pooling -> MLP -> score
    - Video (5D): backbone -> pooling -> transformer -> MLP -> score

    The two modes share backbone and regression head.
    Transformer fusion is only applied to video inputs.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        model_cfg = cfg.model

        self.backbone_name = model_cfg.backbone
        self.dropout_rate = model_cfg.dropout
        self.freeze_backbone = model_cfg.freeze_backbone
        self.num_vqa_layers = model_cfg.transformer_layers or 4
        self.num_frames = model_cfg.num_frames

        # Backbone
        if self.backbone_name == "swin_t":
            swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            self.backbone = swin.features
            self.num_features = swin.head.in_features
        elif self.backbone_name == "resnet50":
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(
                res.conv1,
                res.bn1,
                res.relu,
                res.maxpool,
                res.layer1,
                res.layer2,
                res.layer3,
                res.layer4,
            )
            self.num_features = res.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adaptive pooling: [B, C, H, W] -> [B, C, 1, 1]
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Transformer for video temporal fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_features,
            nhead=8,
            dim_feedforward=self.num_features * 2,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(encoder_layer, num_layers=self.num_vqa_layers)

        # Regression head: 3-layer MLP
        self.quality_head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.quality_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features and pool to a vector."""
        features = self.backbone(x)

        # Handle different output shapes from different backbones
        if features.dim() == 4:
            if features.shape[-1] == self.num_features and features.shape[1] != self.num_features:
                features = features.permute(0, 3, 1, 2)
        elif features.dim() == 3:
            B, L, C = features.shape
            H = W = int(L**0.5)
            if H * W == L:
                features = features.permute(0, 2, 1).view(B, C, H, W)

        pooled = self.spatial_pool(features)  # [B, C, 1, 1]
        return torch.flatten(pooled, 1)       # [B, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        4D input (image):  [B, 3, H, W]   -> score [B]
        5D input (video): [B, F, 3, H, W] -> score [B]
        """
        # Handle grayscale
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.dim() == 5 and x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        if x.dim() == 4:
            # Image path
            features = self._extract_features(x)
            score = self.quality_head(features)
            return score.squeeze(-1)

        elif x.dim() == 5:
            # Video path
            B, F, C, H, W = x.shape

            # Sample frames if needed
            if F != self.num_frames:
                indices = np.linspace(0, F - 1, self.num_frames, dtype=int)
                x = x[:, indices, :, :, :]
                B, F, C, H, W = x.shape

            # Extract features per frame
            x_reshaped = x.view(B * F, C, H, W)
            frame_features = self._extract_features(x_reshaped)  # [B*F, C]

            # Temporal fusion
            v = frame_features.view(B, F, self.num_features)     # [B, F, C]
            v_fused = self.temporal_fusion(v)                    # [B, F, C]
            v_global = torch.mean(v_fused, dim=1)                # [B, C]

            score = self.quality_head(v_global)
            return score.squeeze(-1)

        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")


class IQAVQALoss(nn.Module):
    """
    Loss function with task-aware weighting.

    IQA: MSE + Rank Loss
    VQA: MSE + Rank Loss + PLCC Loss
    """

    def __init__(self, cfg: Config):
        super().__init__()
        loss_cfg = cfg.loss

        self.iqa_mse_weight = loss_cfg.mse_weight
        self.iqa_rank_weight = loss_cfg.rank_weight
        self.iqa_plcc_weight = 0.0

        self.vqa_mse_weight = loss_cfg.mse_weight
        self.vqa_rank_weight = loss_cfg.rank_weight
        self.vqa_plcc_weight = loss_cfg.plcc_weight

        self.max_pairs = loss_cfg.max_pairs
        self.mse_loss = nn.MSELoss()

    def plcc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - PLCC."""
        pred = pred.view(-1)
        target = target.view(-1)

        pred_n = pred - pred.mean()
        target_n = target - target.mean()

        plcc = (pred_n * target_n).sum() / (
            torch.sqrt((pred_n**2).sum()) * torch.sqrt((target_n**2).sum()) + 1e-8
        )
        return 1 - plcc

    def rank_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pairwise rank loss with sampling."""
        pred = pred.view(-1)
        target = target.view(-1)
        n = len(pred)

        if n > self.max_pairs:
            idx = torch.randperm(n)[:self.max_pairs]
            pred = pred[idx]
            target = target[idx]
            n = self.max_pairs

        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1)

        loss = torch.relu(-pred_diff * target_diff)
        return loss.mean() + 1e-8

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: Optional[nn.Module] = None,
        mode: str = "iqa",
    ) -> Dict[str, torch.Tensor]:
        pred = pred.view(-1)
        target = target.view(-1).float()
        pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)

        mse = self.mse_loss(pred, target)
        rank = self.rank_loss(pred, target)
        plcc = self.plcc_loss(pred, target)

        if mode == "iqa":
            total = self.iqa_mse_weight * mse + self.iqa_rank_weight * rank
        else:
            total = self.vqa_mse_weight * mse + self.vqa_rank_weight * rank + self.vqa_plcc_weight * plcc

        return {
            "total_loss": total,
            "mse_loss": mse,
            "rank_loss": rank,
            "plcc_loss": plcc,
            "mode": mode,
        }


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute PLCC, SROCC, RMSE, MAE."""
    import numpy as np
    from scipy.stats import spearmanr

    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    plcc = (pred_n * target_n).sum() / (
        np.sqrt((pred_n**2).sum()) * np.sqrt((target_n**2).sum()) + 1e-8
    )

    srocc, _ = spearmanr(pred, target)
    rmse = np.sqrt(((pred - target) ** 2).mean())
    mae = np.abs(pred - target).mean()

    return {"plcc": float(plcc), "srocc": float(srocc), "rmse": float(rmse), "mae": float(mae)}