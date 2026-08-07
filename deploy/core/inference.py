# deploy/core/inference.py
"""
Core inference logic for IQA/VQA models.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
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


def denormalize(raw_score: float, mos_min: Optional[float], mos_max: Optional[float]) -> Optional[float]:
    if mos_min is None or mos_max is None:
        return None
    return raw_score * (mos_max - mos_min) + mos_min


def image_to_video_tensor(image_tensor: torch.Tensor, num_frames: int = 8) -> torch.Tensor:
    return image_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def sample_frames_from_video(video_path: Path, num_frames: int = 8) -> List[np.ndarray]:
    frames = []

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        else:
            indices = list(range(total_frames))
        frames = vr.get_batch(indices).asnumpy()
        return [f for f in frames]
    except Exception as e:
        logger.debug(f"Decord sampling failed, falling back to OpenCV: {e}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = range(total_frames)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    file_path: Path,
    config: Any,
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> Dict[str, Any]:
    preprocessor = Preprocessor()
    ext = file_path.suffix.lower()
    config_dict = _ensure_dict(config)

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


@torch.no_grad()
def predict_with_resnet_style(
    model: torch.nn.Module,
    file_path: Path,
    config: Any,
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> Dict[str, Any]:
    preprocessor = Preprocessor()
    ext = file_path.suffix.lower()
    config_dict = _ensure_dict(config)

    if ext in cfg.image_exts:
        data_tensor = preprocessor.process_image(file_path).unsqueeze(0).to(device)

        with torch.no_grad():
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
            "task_type": config_dict.get("task_type", "iqa"),
            "model_name": config_dict.get("model", {}).get("name", "ResNet-style"),
        }

    else:
        frames = sample_frames_from_video(file_path, cfg.num_frames)
        if not frames:
            raise ValueError(f"Failed to sample frames from video: {file_path}")

        scores = []
        for frame_np in frames:
            tensor = preprocessor.process_image_from_array(frame_np).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor).float()
                if out.ndim > 1 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                scores.append(float(out.flatten()[0].cpu().item()))

        avg_raw = sum(scores) / len(scores)

        dataset_cfg = _get_dataset_config(config_dict)
        mos_min = mos_min if mos_min is not None else dataset_cfg.get("mos_min")
        mos_max = mos_max if mos_max is not None else dataset_cfg.get("mos_max")
        real_score = denormalize(avg_raw, mos_min, mos_max)

        return {
            "file": str(file_path),
            "raw_score": round(avg_raw, 6),
            "mos_score": round(real_score, 4) if real_score is not None else None,
            "task_type": config_dict.get("task_type", "iqa"),
            "model_name": config_dict.get("model", {}).get("name", "ResNet-style"),
        }
