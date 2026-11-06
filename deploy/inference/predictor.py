"""
Core inference logic for IQA/VQA models.
"""

from pathlib import Path
from typing import List, Optional, TypedDict

import torch
from loguru import logger
from pydantic import BaseModel

from ..config import cfg
from .preprocessor import Preprocessor


class PredictionResult(TypedDict):
    file: str
    raw_score: float
    mos_score: float | None
    task_type: str
    model_name: str


class PredictionError(TypedDict):
    file: str
    error: str


def _ensure_dict(config: object) -> dict[str, object]:
    """Accept a Pydantic model or string-keyed mapping, validating the boundary."""
    if isinstance(config, BaseModel):
        config = config.model_dump()
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise TypeError(f"Expected a string-keyed configuration mapping, got {type(config).__name__}")
    return {key: value for key, value in config.items()}


def _get_dataset_config(config_dict: dict[str, object]) -> dict[str, object]:
    """Support current and legacy dataset config keys."""
    dataset_cfg = config_dict.get("dataset")
    if dataset_cfg is None:
        dataset_cfg = config_dict.get("dataset_info")
    return _ensure_dict({} if dataset_cfg is None else dataset_cfg)


def _get_model_config(config_dict: dict[str, object]) -> dict[str, object]:
    """Return the model configuration mapping from a config dictionary."""
    model_cfg = config_dict.get("model")
    return _ensure_dict({} if model_cfg is None else model_cfg)


def _integer_setting(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _optional_mos(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("MOS bounds must be numbers or None")
    return float(value)


def _text_setting(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def denormalize_to_mos(
    raw_score: float,
    mos_min: Optional[float],
    mos_max: Optional[float],
) -> Optional[float]:
    """Map a normalized score back to the dataset MOS range, if one is known."""
    if mos_min is None or mos_max is None:
        return None
    return raw_score * (mos_max - mos_min) + mos_min


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    file_path: Path,
    config: object,
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> PredictionResult:
    """Run inference on one image or video and return the measured scores."""
    ext = file_path.suffix.lower()
    config_dict = _ensure_dict(config)
    model_cfg = _get_model_config(config_dict)
    preprocessor = Preprocessor(
        num_frames=_integer_setting(model_cfg.get("num_frames", cfg.num_frames), "num_frames"),
        input_size=_integer_setting(model_cfg.get("input_size", cfg.input_size), "input_size"),
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
    mos_min = mos_min if mos_min is not None else _optional_mos(dataset_cfg.get("mos_min"))
    mos_max = mos_max if mos_max is not None else _optional_mos(dataset_cfg.get("mos_max"))

    real_score = denormalize_to_mos(raw_score, mos_min, mos_max)

    return {
        "file": str(file_path),
        "raw_score": round(raw_score, 6),
        "mos_score": round(real_score, 4) if real_score is not None else None,
        "task_type": _text_setting(config_dict.get("task_type", "unknown"), "task_type"),
        "model_name": _text_setting(model_cfg.get("name", "unknown"), "model.name"),
    }


def predict_batch(
    model: torch.nn.Module,
    file_paths: List[Path],
    config: object,
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> List[PredictionResult | PredictionError]:
    """Run inference on each path, returning one result (or error) per file."""
    results: list[PredictionResult | PredictionError] = []
    for f in file_paths:
        try:
            results.append(predict_single(model, f, config, device, mos_min, mos_max))
        except Exception as e:
            logger.error(f"[ERROR] Inference failed: {f} | {e}")
            results.append({"file": str(f), "error": str(e)})
    return results
