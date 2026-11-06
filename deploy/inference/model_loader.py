"""Serving checkpoint selection and the shared safe model loader."""

from pathlib import Path

from loguru import logger

from src.utils.checkpointing import load_checkpoint

__all__ = ["load_checkpoint", "select_checkpoint"]


def select_checkpoint(targets: list, cfg) -> Path:
    """
    Auto-select checkpoint based on file types.

    Args:
        targets: List of file paths
        cfg: InferenceConfig instance

    Returns:
        Path to the selected checkpoint
    """
    from ..config import cfg as config_global

    has_image = any(p.suffix.lower() in config_global.image_exts for p in targets)
    has_video = any(p.suffix.lower() in config_global.video_exts for p in targets)

    if has_image and not has_video:
        logger.info("[INFO] Auto-selected IQA model for image files")
        return Path(cfg.iqa_model_path)
    elif has_video and not has_image:
        logger.info("[INFO] Auto-selected VQA model for video files")
        return Path(cfg.vqa_model_path)
    else:
        if has_image and has_video:
            logger.warning("[WARN] Mixed image and video files, using IQA model")
        else:
            logger.warning("[WARN] No supported files found, using IQA model")
        return Path(cfg.iqa_model_path)
