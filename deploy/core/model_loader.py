# deploy/core/model_loader.py
"""
Model loading utilities for inference.
"""

import torch
from pathlib import Path
from typing import Tuple
from loguru import logger

from src.models.iqavqa_net import IQAVQANet
from src.config.schemas import Config


def load_checkpoint(
    ckpt_path: Path,
    device: str = "cuda"
) -> Tuple[torch.nn.Module, Config]:
    """
    Load model checkpoint saved by TrainerEngine.

    Args:
        ckpt_path: Path to the .pt checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, config_obj)

    Raises:
        FileNotFoundError: If checkpoint does not exist
        KeyError: If checkpoint lacks required config field
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint missing 'config' field. "
            "Ensure checkpoint was saved by TrainerEngine._save_checkpoint"
        )

    config_dict = checkpoint["config"]

    # Convert dict to Config object
    if isinstance(config_dict, dict):
        config_obj = Config(**config_dict)
    elif isinstance(config_dict, Config):
        config_obj = config_dict
    else:
        raise TypeError(f"Unexpected config type: {type(config_dict)}")

    model = IQAVQANet(cfg=config_obj, load_pretrained_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    logger.info(f"[OK] Loaded: {ckpt_path.name} (epoch={epoch})")

    if metrics:
        val_srocc = metrics.get("val_srocc") or metrics.get("srocc")
        if val_srocc is not None:
            logger.info(f"   └─ Validation SROCC: {val_srocc:.4f}")

    return model, config_obj




def select_checkpoint(targets: list, cfg) -> Path:
    """
    Auto-select checkpoint based on file types.

    Args:
        targets: List of file paths
        cfg: InferenceConfig instance

    Returns:
        Path to the selected checkpoint
    """
    from .runtime_config import cfg as config_global

    has_image = any(p.suffix.lower() in config_global.image_exts for p in targets)
    has_video = any(p.suffix.lower() in config_global.video_exts for p in targets)

    if has_image and not has_video:
        logger.info("[INFO] Auto-selected IQA model for image files")
        return cfg.iqa_model_path
    elif has_video and not has_image:
        logger.info("[INFO] Auto-selected VQA model for video files")
        return cfg.vqa_model_path
    else:
        if has_image and has_video:
            logger.warning("[WARN] Mixed image and video files, using IQA model")
        else:
            logger.warning("[WARN] No supported files found, using IQA model")
        return cfg.iqa_model_path
