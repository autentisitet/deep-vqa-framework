# src/models/iqavqa_model.py
import math
from typing import Optional

import torch
import torch.nn as nn

from src.config.schemas import Config
from .backbones import build_backbone, known_feature_dim
from .heads import build_quality_head, init_head_weights
from .torchvision_cache import configure_torch_weight_cache


class IQAVQANet(nn.Module):
    """
    Quality assessment network supporting both images and videos.

    - Image (4D): Swin-T ImageNet backbone -> pooling -> MLP -> score
    - Video (5D): Swin-T ImageNet backbone -> pooling -> transformer -> MLP -> score

    The forward path is selected from tensor dimensionality:
    - [B, C, H, W] for images
    - [B, F, C, H, W] for videos
    """

    def __init__(self, cfg: Config, load_pretrained_backbone: Optional[bool] = None):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        use_pretrained = model_cfg.pretrained if load_pretrained_backbone is None else load_pretrained_backbone
        self._load_pretrained_backbone = bool(use_pretrained)
        if self._load_pretrained_backbone:
            configure_torch_weight_cache(cfg)

        self.configured_backbone_name = model_cfg.backbone
        self.image_backbone_name = getattr(model_cfg, "image_backbone", None) or self.configured_backbone_name
        self.video_backbone_name = getattr(model_cfg, "video_backbone", "swin_t")
        self.dropout_rate = model_cfg.dropout
        self.freeze_backbone = model_cfg.freeze_backbone
        self.num_vqa_layers = model_cfg.transformer_layers or 4
        self.num_frames = model_cfg.num_frames
        self.target_media_type = self._infer_target_media_type(cfg)

        self.image_num_features = known_feature_dim(self.image_backbone_name)
        self.video_num_features = known_feature_dim(self.video_backbone_name)
        self.num_features = self.image_num_features if self.target_media_type == "image" else self.video_num_features

        self.image_backbone: Optional[nn.Module] = None
        self.video_backbone: Optional[nn.Module] = None
        self.image_feature_norm: Optional[nn.Module] = None
        self.video_feature_norm: Optional[nn.Module] = None
        self.temporal_fusion: Optional[nn.Module] = None
        self.temporal_position_embedding: Optional[nn.Parameter] = None
        self.image_quality_head: Optional[nn.Sequential] = None
        self.video_quality_head: Optional[nn.Sequential] = None

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        if self.target_media_type == "image":
            self._ensure_image_branch(use_pretrained)
        elif self.target_media_type == "video":
            self._ensure_video_branch(use_pretrained)
        else:
            self._ensure_image_branch(use_pretrained)
            self._ensure_video_branch(use_pretrained)

    def _infer_target_media_type(self, cfg: Config) -> str:
        data_type = getattr(cfg.dataset, "data_type", "").lower()
        if data_type in {"image", "video"}:
            return data_type

        task_type = getattr(cfg, "task_type", "").lower()
        if task_type == "iqa":
            return "image"
        if task_type == "vqa":
            return "video"

        if self.configured_backbone_name == self.image_backbone_name:
            return "image"
        if self.configured_backbone_name == self.video_backbone_name:
            return "video"
        return "both"

    def _freeze_module_if_needed(self, module: nn.Module) -> None:
        if not self.freeze_backbone:
            return
        for param in module.parameters():
            param.requires_grad = False

    def _current_model_device(self) -> torch.device | None:
        for tensor in self.parameters():
            return tensor.device
        for tensor in self.buffers():
            return tensor.device
        return None

    def _move_modules_to_device(self, device: torch.device | None, *modules: Optional[nn.Module]) -> None:
        if device is None:
            return
        for module in modules:
            if module is None:
                continue
            module.to(device)

    def _build_temporal_fusion(self) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.video_num_features,
            nhead=8,
            dim_feedforward=self.video_num_features * 2,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=self.num_vqa_layers)

    def _build_temporal_position_embedding(self, device: torch.device | None = None) -> nn.Parameter:
        position = torch.arange(self.num_frames, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.video_num_features, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / self.video_num_features)
        )

        pe = torch.zeros(1, self.num_frames, self.video_num_features, dtype=torch.float32, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
        return nn.Parameter(pe)

    def _position_embedding_for(self, frame_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.temporal_position_embedding is None:
            self.temporal_position_embedding = self._build_temporal_position_embedding(device=device)

        pos = self.temporal_position_embedding
        if pos.device != device:
            pos = pos.to(device)

        if pos.shape[1] == frame_count:
            return pos.to(dtype=dtype)

        pos_resampled = torch.nn.functional.interpolate(
            pos.permute(0, 2, 1),
            size=frame_count,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        return pos_resampled.to(dtype=dtype)

    def _ensure_image_branch(self, use_pretrained: Optional[bool] = None) -> None:
        if self.image_backbone is not None and self.image_quality_head is not None:
            return

        load_pretrained = self._load_pretrained_backbone if use_pretrained is None else bool(use_pretrained)
        target_device = self._current_model_device()
        self.image_backbone, self.image_num_features, self.image_feature_norm = build_backbone(
            self.image_backbone_name,
            load_pretrained,
        )
        self._freeze_module_if_needed(self.image_backbone)
        if self.image_feature_norm is not None:
            self._freeze_module_if_needed(self.image_feature_norm)
        self.image_quality_head = build_quality_head(self.image_num_features, self.dropout_rate)
        init_head_weights(self.image_quality_head)
        self._move_modules_to_device(
            target_device,
            self.image_backbone,
            self.image_feature_norm,
            self.image_quality_head,
        )

    def _ensure_video_branch(self, use_pretrained: Optional[bool] = None) -> None:
        if (
            self.video_backbone is not None
            and self.temporal_fusion is not None
            and self.video_quality_head is not None
        ):
            return

        load_pretrained = self._load_pretrained_backbone if use_pretrained is None else bool(use_pretrained)
        target_device = self._current_model_device()
        self.video_backbone, self.video_num_features, self.video_feature_norm = build_backbone(
            self.video_backbone_name,
            load_pretrained,
        )
        self._freeze_module_if_needed(self.video_backbone)
        if self.video_feature_norm is not None:
            self._freeze_module_if_needed(self.video_feature_norm)
        self.temporal_fusion = self._build_temporal_fusion()
        self.temporal_position_embedding = self._build_temporal_position_embedding(device=target_device)
        self.video_quality_head = build_quality_head(self.video_num_features, self.dropout_rate)
        init_head_weights(self.video_quality_head)
        self._move_modules_to_device(
            target_device,
            self.video_backbone,
            self.video_feature_norm,
            self.temporal_fusion,
            self.video_quality_head,
        )

    def _extract_features(
        self,
        x: torch.Tensor,
        backbone: nn.Module,
        num_features: int,
        feature_norm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Extract backbone features and pool to a vector."""
        features = backbone(x)

        if feature_norm is not None:
            if features.dim() in {3, 4} and features.shape[-1] == num_features:
                features = feature_norm(features)
            else:
                raise ValueError(
                    f"Feature norm expects channel-last features with {num_features} channels, "
                    f"got shape={tuple(features.shape)}"
                )

        if features.dim() == 4:
            if features.shape[-1] == num_features and features.shape[1] != num_features:
                features = features.permute(0, 3, 1, 2)
        elif features.dim() == 3:
            B, L, C = features.shape
            H = W = int(L**0.5)
            if H * W == L:
                features = features.permute(0, 2, 1).view(B, C, H, W)

        pooled = self.spatial_pool(features)
        return torch.flatten(pooled, 1)

    def extract_quality_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the representation consumed by the quality regression head.

        This keeps offline feature-distribution analysis on the exact same
        embedding path as trained IQA/VQA inference.
        """
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.dim() == 5 and x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        if x.dim() == 4:
            if x.shape[1] != 3:
                raise ValueError(f"Expected image input [B, 3, H, W], got shape={tuple(x.shape)}")
            self._ensure_image_branch()
            if self.image_backbone is None:
                raise RuntimeError("Image branch was not initialized.")
            return self._extract_features(
                x, self.image_backbone, self.image_num_features, self.image_feature_norm
            )

        if x.dim() == 5:
            if x.shape[2] != 3:
                raise ValueError(f"Expected video input [B, F, 3, H, W], got shape={tuple(x.shape)}")
            self._ensure_video_branch()
            if self.video_backbone is None or self.temporal_fusion is None:
                raise RuntimeError("Video branch was not initialized.")
            batch, frames, channels, height, width = x.shape
            if frames < 1:
                raise ValueError(f"Expected at least 1 video frame, got {frames}")
            if frames != self.num_frames:
                indices = torch.linspace(0, frames - 1, self.num_frames, device=x.device).long()
                x = x[:, indices, :, :, :]
                batch, frames, channels, height, width = x.shape

            frame_features = self._extract_features(
                x.reshape(batch * frames, channels, height, width),
                self.video_backbone,
                self.video_num_features,
                self.video_feature_norm,
            ).view(batch, frames, self.video_num_features)
            frame_features = frame_features + self._position_embedding_for(
                frames, device=frame_features.device, dtype=frame_features.dtype
            )
            return torch.mean(self.temporal_fusion(frame_features), dim=1)

        raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        4D input (image):  [B, 3, H, W]   -> score [B]
        5D input (video): [B, F, 3, H, W] -> score [B]
        """
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.dim() == 5 and x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        if x.dim() == 4:
            if x.shape[1] != 3:
                raise ValueError(f"Expected image input [B, 3, H, W], got shape={tuple(x.shape)}")
            self._ensure_image_branch()
            if self.image_backbone is None or self.image_quality_head is None:
                raise RuntimeError("Image branch was not initialized.")
            features = self._extract_features(
                x,
                self.image_backbone,
                self.image_num_features,
                self.image_feature_norm,
            )
            score = self.image_quality_head(features)
            return score.squeeze(-1)

        if x.dim() == 5:
            if x.shape[2] != 3:
                raise ValueError(f"Expected video input [B, F, 3, H, W], got shape={tuple(x.shape)}")
            self._ensure_video_branch()
            if self.video_backbone is None or self.temporal_fusion is None or self.video_quality_head is None:
                raise RuntimeError("Video branch was not initialized.")
            B, F, C, H, W = x.shape
            if F < 1:
                raise ValueError(f"Expected at least 1 video frame, got shape={tuple(x.shape)}")

            if F != self.num_frames:
                indices = torch.linspace(0, F - 1, self.num_frames, device=x.device).long()
                x = x[:, indices, :, :, :]
                B, F, C, H, W = x.shape

            x_reshaped = x.reshape(B * F, C, H, W)
            frame_features = self._extract_features(
                x_reshaped,
                self.video_backbone,
                self.video_num_features,
                self.video_feature_norm,
            )

            v = frame_features.view(B, F, self.video_num_features)
            v = v + self._position_embedding_for(F, device=v.device, dtype=v.dtype)
            v_fused = self.temporal_fusion(v)
            v_global = torch.mean(v_fused, dim=1)

            score = self.video_quality_head(v_global)
            return score.squeeze(-1)

        raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")


__all__ = ["IQAVQANet"]
