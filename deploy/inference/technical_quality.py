"""Deterministic handcrafted image/video quality heuristic for comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader, cpu

from src.data.image_io import read_image_rgb


@dataclass(frozen=True)
class TechnicalQuality:
    score: int
    description: str
    frames_sampled: int
    sharpness: float
    noise: float
    exposure: float
    contrast: float
    temporal_stability: float


def _clamp_to_range(
    value: float,
    low: float = 0.0,
    high: float = 100.0,
) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return max(low, min(high, float(value)))


def _frame_quality(frame: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(sharpness, noise, exposure, contrast)`` for one RGB frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    mean = float(normalized.mean())
    dark = float((normalized <= 0.03).mean())
    bright = float((normalized >= 0.97).mean())
    contrast = _clamp_to_range(float(normalized.std()) * 360.0)

    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = _clamp_to_range(100.0 * (1.0 - np.exp(-laplacian / 180.0)))

    median = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - median.astype(np.float32)
    noise_level = float(np.std(residual))
    noise = _clamp_to_range(100.0 - noise_level * 7.0)

    clipping = max(0.0, dark + bright - 0.25)
    exposure = _clamp_to_range(100.0 - clipping * 300.0 - abs(mean - 0.5) * 45.0)
    if dark > 0.995 or bright > 0.995:
        exposure = 0.0
        contrast = 0.0
    return sharpness, noise, exposure, contrast


def _read_image(path: Path) -> list[np.ndarray]:
    """Read one image and return it as a single-frame list."""
    return [read_image_rgb(path)]


def _read_video(path: Path, max_frames: int) -> list[np.ndarray]:
    """Sample up to ``max_frames`` evenly spaced frames from a video."""
    vr = VideoReader(str(path), ctx=cpu(0))
    frame_count = max(1, len(vr))
    sample_count = min(max_frames, frame_count)
    positions = sorted({round(index * (frame_count - 1) / max(sample_count - 1, 1)) for index in range(sample_count)})
    frames = [vr[position].asnumpy() for position in positions]
    if not frames:
        raise ValueError(f"No decodable frames in video: {path}")
    return frames


def _description(
    score: int,
    sharpness: float,
    noise: float,
    exposure: float,
    stability: float,
) -> str:
    """Build a short human-readable summary of the measured quality."""
    observations: list[str] = []
    observations.append("sharp" if sharpness >= 70 else "slightly soft" if sharpness >= 35 else "blurred")
    observations.append("low noise" if noise >= 75 else "visible noise" if noise >= 45 else "heavy noise")
    if exposure < 45:
        observations.append("poor exposure")
    elif exposure < 75:
        observations.append("uneven exposure")
    if stability < 55:
        observations.append("inconsistent frames")
    return ", ".join(observations)[:50]


def assess_technical_quality(
    path: Path,
    media_type: str,
    max_video_frames: int = 4,
) -> TechnicalQuality:
    """Apply a reproducible heuristic without claiming calibrated perceptual quality."""
    frames = _read_image(path) if media_type == "image" else _read_video(path, max_video_frames)
    metrics = np.asarray([_frame_quality(frame) for frame in frames], dtype=np.float32)
    sharpness, noise, exposure, contrast = metrics.mean(axis=0).tolist()
    stability = 100.0
    if len(frames) > 1:
        resized = [cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) for frame in frames]
        differences = [float(np.mean(np.abs(current - previous))) for previous, current in zip(resized, resized[1:])]
        stability = _clamp_to_range(100.0 - np.mean(differences) * 1.8)

    score_value = (
        sharpness * 0.35 + noise * 0.25 + exposure * 0.20 + contrast * 0.20 - max(0.0, 60.0 - stability) * 0.25
    )
    if noise < 20.0:
        score_value *= 0.65
    if exposure < 20.0:
        score_value *= 0.55
    score = round(_clamp_to_range(score_value))
    description = _description(score, sharpness, noise, exposure, stability)
    return TechnicalQuality(
        score=score,
        description=description,
        frames_sampled=len(frames),
        sharpness=round(sharpness, 2),
        noise=round(noise, 2),
        exposure=round(exposure, 2),
        contrast=round(contrast, 2),
        temporal_stability=round(stability, 2),
    )


__all__ = ["TechnicalQuality", "assess_technical_quality"]
