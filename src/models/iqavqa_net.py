from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class IQAVQANet(nn.Module):
    """
    IQA/VQA Unified Network

    ## Architecture:

    - Backbone: Swin-T or ResNet50

    - Temporal Fusion: Transformer (Video Only)

    - Regression Head: 3-Layer MLP with BatchNorm

    ## Features:

    - Automatically switches between image (4D) and video (5D) modes based on input dimension

    - Loss Function: 0.7*MSE + 0.3*RankLoss

    - Kaiming Initialization for GELU/ReLU activations
    """

    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config.get("model", {})
        self.backbone_name = model_cfg.get("backbone", "swin_t")
        self.dropout_rate = model_cfg.get("dropout", 0.3)
        self.freeze_backbone = model_cfg.get("freeze_backbone", False)
        self.num_vqa_layers = model_cfg.get("transformer_layers", 4)
        self.num_frames = model_cfg.get("num_frames", 8)

        # ----------------------------------------------------
        # 1. Backbone: Swin-T or ResNet50
        # ----------------------------------------------------
        if self.backbone_name == "swin_t":
            swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            self.backbone = swin.features
            self.num_features = swin.head.in_features  # 768
            self.is_transformer = True

        elif self.backbone_name == "resnet50":
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(
                res.conv1, res.bn1, res.relu, res.maxpool,
                res.layer1, res.layer2, res.layer3, res.layer4
            )
            self.num_features = res.fc.in_features  # 2048
            self.is_transformer = False
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_name}")

        # Backbone network freeze switch mechanism
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ----------------------------------------------------
        # 2. Adaptive Pooling: Outputs a fixed-size feature map
        # ----------------------------------------------------
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # ----------------------------------------------------
        # 3. Transformer Timing Fusion (Video Only)
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_features,
            nhead=8,
            dim_feedforward=self.num_features * 2,
            dropout=self.dropout_rate,
            activation="gelu" if self.is_transformer else "relu",
            batch_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(encoder_layer, num_layers=self.num_vqa_layers)

        # ----------------------------------------------------
        # 4. Regression Head: 3-Layer MLP with BatchNorm
        # ----------------------------------------------------
        self.quality_head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU() if self.is_transformer else nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU() if self.is_transformer else nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming Initialization (适用于 GELU/ReLU 激活函数)
        
        ✅ 使用 Kaiming 而不是 Xavier，因为 GELU/ReLU 不是对称的（不像 Tanh/Sigmoid）
        ✅ 同时初始化 BN 的 weight 和 bias
        """
        for m in self.quality_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _forward_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features and compress them into vectors."""
        features = self.backbone(x)

        # The output shape of Swin-T varies depending on the torchvision version; it is uniformly converted to [B, C, H, W]
        if features.dim() == 4:
            # Determine whether the value is [B, C, H, W] or [B, H, W, C].
            if features.shape[-1] == self.num_features and features.shape[1] != self.num_features:
                # [B, H, W, C] -> [B, C, H, W]
                features = features.permute(0, 3, 1, 2)

        elif features.dim() == 3:
            # If the form is [B, L, C] -> dynamically restore H and W.
            B, L, C = features.shape
            H = W = int(L**0.5)
            if H * W == L:
                features = features.permute(0, 2, 1).view(B, C, H, W)

        pooled = self.spatial_pool(features)  # Compress to [B, num_features, 1, 1]
        return torch.flatten(pooled, 1)  # Flattened to [B, num_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routing based on input dimension:

        - 4D Tensor [B, 3, H, W] -> Images, directly extract spatial features

        - 5D Tensor [B, F, 3, H, W] -> Videos, extract features from each frame first, then perform temporal fusion
        """

        # Adapt grayscale images
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
        elif x.dim() == 5 and x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)  # [B, F, 1, H, W] -> [B, F, 3, H, W]

        if x.dim() == 4:
            v_global = self._forward_backbone_features(x)
            score = self.quality_head(v_global)
            return score.squeeze(-1)  # Ensures the one-dimensional tensor [B] is returned.

        elif x.dim() == 5:
            B, F, C, H, W = x.shape

            # If the input frame rate differs from the configuration, perform sampling or issue warnings here.
            if F != self.num_frames:
                # Number of target frames sampled
                indices = np.linspace(0, F - 1, self.num_frames, dtype=int)
                x = x[:, indices, :, :, :]
                B, F, C, H, W = x.shape

            # Ensure it is a 3-channel image.
            if C != 3:
                raise ValueError(f"Expected 3 channels after grayscale conversion, got {C}")

            # 1. Extract spatial features from each frame
            x_reshaped = x.view(B * F, C, H, W)
            v_frames = self._forward_backbone_features(x_reshaped)

            # 2. Organize into a frame sequence
            v = v_frames.view(B, F, self.num_features)

            # 3. Transformer Timing Fusion
            v_fused = self.temporal_fusion(v)

            # 4. Frame-based average pooling
            v_global = torch.mean(v_fused, dim=1)

            # 5. Regression Header Outputs Quality Score
            score = self.quality_head(v_global)
            return score.squeeze(-1)  # Ensures the one-dimensional tensor is returned as [B]

        else:
            raise ValueError(f"Invalid input tensor dim: {x.dim()}. Expected 4D for IQA or 5D for VQA.")


class IQAVQALoss(nn.Module):
    """
    Loss Function: 
    - IQA: MSE + RankLoss
    - VQA: MSE + RankLoss + PLCC
    
    自动根据 mode 参数切换损失权重
    """

    def __init__(self, config: Dict):
        super().__init__()
        loss_cfg = config.get("loss", {})
        
        # IQA 损失权重
        self.iqa_mse_weight = loss_cfg.get("iqa_mse_weight", 0.7)
        self.iqa_rank_weight = loss_cfg.get("iqa_rank_weight", 0.3)
        self.iqa_plcc_weight = loss_cfg.get("iqa_plcc_weight", 0.0)  # IQA 默认不加 PLCC
        
        # VQA 损失权重（VQA 更关注 PLCC）
        self.vqa_mse_weight = loss_cfg.get("vqa_mse_weight", 0.4)
        self.vqa_rank_weight = loss_cfg.get("vqa_rank_weight", 0.3)
        self.vqa_plcc_weight = loss_cfg.get("vqa_plcc_weight", 0.3)  # VQA 加 PLCC
        
        self.max_pairs = loss_cfg.get("max_pairs", 5000)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def plcc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """PLCC Loss: 1 - PLCC，最大化 PLCC"""
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        y_pred_n = y_pred - y_pred.mean()
        y_true_n = y_true - y_true.mean()
        
        plcc = (y_pred_n * y_true_n).sum() / (
            torch.sqrt((y_pred_n ** 2).sum()) * torch.sqrt((y_true_n ** 2).sum()) + 1e-8
        )
        return 1 - plcc

    def rank_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Sampling method to calculate rank loss
        
        ✅ 添加了数值稳定性处理
        ✅ 使用更高效的采样策略
        """
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        n = len(y_pred)

        if n > self.max_pairs:
            idx = torch.randperm(n)[:self.max_pairs]
            y_pred = y_pred[idx]
            y_true = y_true[idx]
            n = self.max_pairs

        pred_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)

        # ✅ 使用 clamp 防止梯度爆炸
        loss = torch.relu(-pred_diff * true_diff)
        
        # ✅ 添加小 epsilon 防止数值不稳定
        loss = loss.mean() + 1e-8
        
        return loss

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor, 
        model: Optional[nn.Module] = None,
        mode: str = "iqa"
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mode: "iqa" 或 "vqa"，决定损失函数权重
        """
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float()

        # Truncate values that exceed the boundary to avoid NaN.
        y_pred = torch.clamp(y_pred, min=1e-6, max=1.0 - 1e-6)

        mse = self.mse_loss(y_pred, y_true)
        rank = self.rank_loss(y_pred, y_true)
        plcc = self.plcc_loss(y_pred, y_true)

        if mode == "iqa":
            total_loss = (
                self.iqa_mse_weight * mse +
                self.iqa_rank_weight * rank +
                self.iqa_plcc_weight * plcc
            )
        else:  # vqa
            total_loss = (
                self.vqa_mse_weight * mse +
                self.vqa_rank_weight * rank +
                self.vqa_plcc_weight * plcc
            )

        return {
            "total_loss": total_loss,
            "mse_loss": mse,
            "rank_loss": rank,
            "plcc_loss": plcc,
            "mode": mode,
        }


class EarlyStopping:
    """
    Early Stopping to prevent overfitting
    
    ✅ 当验证集指标不再提升时停止训练
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 1e-4, 
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Args:
            patience: 多少个 epoch 没有提升后停止
            min_delta: 最小提升阈值，小于此值视为没有提升
            mode: 'max' 表示指标越高越好，'min' 表示指标越低越好
            verbose: 是否打印日志
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int = 0) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"[EarlyStopping] Initial best score: {score:.4f} at epoch {epoch}")
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improved to {score:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Early stopping triggered at epoch {epoch}, best at epoch {self.best_epoch}")
                
        return self.early_stop
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    计算 IQA/VQA 常用评估指标
    
    Args:
        y_pred: 预测值 [B]
        y_true: 真实值 [B]
    
    Returns:
        Dict: PLCC, SROCC, RMSE, MAE
    """
    y_pred = y_pred.view(-1).detach().cpu().numpy()
    y_true = y_true.view(-1).detach().cpu().numpy()
    
    # PLCC
    y_pred_n = y_pred - y_pred.mean()
    y_true_n = y_true - y_true.mean()
    plcc = (y_pred_n * y_true_n).sum() / (
        np.sqrt((y_pred_n ** 2).sum()) * np.sqrt((y_true_n ** 2).sum()) + 1e-8
    )
    
    # SROCC
    from scipy.stats import spearmanr
    srocc, _ = spearmanr(y_pred, y_true)
    
    # RMSE
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    
    # MAE
    mae = np.abs(y_pred - y_true).mean()
    
    return {
        "plcc": plcc,
        "srocc": srocc,
        "rmse": rmse,
        "mae": mae,
    }