# deploy/core/inference.py
"""
Core inference logic for IQA/VQA models.
"""

import torch
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from .runtime_config import cfg
from .preprocessor import Preprocessor


def _ensure_dict(config: Any) -> Dict[str, Any]:
    """Convert Config object to dict if needed."""
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return config


def _get_dataset_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Support current and legacy dataset config keys."""
    dataset_cfg = config_dict.get("dataset") or config_dict.get("dataset_info") or {}
    return dataset_cfg if isinstance(dataset_cfg, dict) else {}


def _get_model_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config_dict.get("model") or {}
    return model_cfg if isinstance(model_cfg, dict) else {}


def denormalize(raw_score: float, mos_min: Optional[float], mos_max: Optional[float]) -> Optional[float]:
    if mos_min is None or mos_max is None:
        return None
    return raw_score * (mos_max - mos_min) + mos_min


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    file_path: Path,
    config: Any,
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> Dict[str, Any]:
    ext = file_path.suffix.lower()
    config_dict = _ensure_dict(config)
    model_cfg = _get_model_config(config_dict)
    preprocessor = Preprocessor(
        num_frames=model_cfg.get("num_frames", cfg.num_frames),
        input_size=model_cfg.get("input_size", cfg.input_size),
    )

    if ext in cfg.image_exts:
        data_tensor = preprocessor.process_image(file_path).unsqueeze(0).to(device)
    elif ext in cfg.video_exts:
        data_tensor = preprocessor.process_video(file_path).unsqueeze(0).to(device)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    output = model(data_tensor).float()
    if output.ndim > 1 and output.size(-1) == 1:
        output = output.squeeze(-1)
    raw_score = float(output.flatten()[0].cpu().item())

    dataset_cfg = _get_dataset_config(config_dict)
    mos_min = mos_min if mos_min is not None else dataset_cfg.get("mos_min")
    mos_max = mos_max if mos_max is not None else dataset_cfg.get("mos_max")

    real_score = denormalize(raw_score, mos_min, mos_max)

    return {
        "file": str(file_path),
        "raw_score": round(raw_score, 6),
        "mos_score": round(real_score, 4) if real_score is not None else None,
        "task_type": config_dict.get("task_type", "unknown"),
        "model_name": config_dict.get("model", {}).get("name", "unknown"),
    }


def predict_batch(
    model: torch.nn.Module,
    file_paths: List[Path],
    config: Any,
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    results = []
    for f in file_paths:
        try:
            results.append(predict_single(model, f, config, device, mos_min, mos_max))
        except Exception as e:
            logger.error(f"[ERROR] Inference failed: {f} | {e}")
            results.append({"file": str(f), "error": str(e)})
    return results
