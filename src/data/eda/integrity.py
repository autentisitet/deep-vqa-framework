# src/data/eda/integrity.py
from pathlib import Path
from typing import Optional, Tuple, TypedDict

import av
import numpy as np
from decord import VideoReader, cpu

from src.data.dataset_types import DatasetType
from src.data.image_io import read_image_rgb


class VideoDiagnostics(TypedDict):
    frame_count: int
    bad_frames: int
    black_frames: int
    white_frames: int
    actual_frames: int
    fps: float
    resolution: tuple[int, int]
    duration_sec: float
    frame_drops: bool
    irregular_interval_ratio: float


def _empty_video_diagnostics() -> VideoDiagnostics:
    """Return the base diagnostics dict shared by every video check."""
    return {
        "frame_count": 0,
        "bad_frames": 0,
        "black_frames": 0,
        "white_frames": 0,
        "actual_frames": 0,
        "fps": 0,
        "resolution": (0, 0),
        "duration_sec": 0,
        "frame_drops": False,
        "irregular_interval_ratio": 0.0,
    }


def _finalize_frame_diagnostics(timestamps: list[float], diagnostics: VideoDiagnostics) -> None:
    """Fill frame-drop fields from the sampled timestamps (in milliseconds)."""
    if len(timestamps) <= 2:
        return
    intervals = np.diff(timestamps)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    diagnostics["irregular_interval_ratio"] = float(std_interval / (mean_interval + 1e-6))
    if np.max(intervals) > mean_interval * 1.5:
        diagnostics["frame_drops"] = True


def _bad_frame_ratio_message(diagnostics: VideoDiagnostics, sample_count: int) -> Optional[str]:
    """Return a failure message when too many sampled frames are bad."""
    bad_ratio = diagnostics["bad_frames"] / max(1, sample_count)
    if bad_ratio > 0.3:
        return f"Too high percentage of bad frames: {bad_ratio:.2%}"
    return None


def _accumulate_luminance_flags(gray: np.ndarray, diagnostics: VideoDiagnostics) -> None:
    """Count near-black and near-white frames from a grayscale frame."""
    mean_gray = float(np.mean(gray))
    if mean_gray < 8.0:
        diagnostics["black_frames"] += 1
    if mean_gray > 245.0:
        diagnostics["white_frames"] += 1


def _accumulate_bad_frame(diff: float, diagnostics: VideoDiagnostics) -> None:
    """Count a frame as bad when it barely differs from the previous sample."""
    if diff < 0.5:
        diagnostics["bad_frames"] += 1


def _decord_timestamp_ms(vr: VideoReader, idx: int, fps: float) -> float:
    """Return the frame timestamp in milliseconds, falling back to frame index."""
    try:
        return float(vr.get_frame_timestamp(idx)[0] * 1000)
    except Exception:
        return idx / fps * 1000


def check_video_integrity(
    path: Path, sample_interval: int = 30
) -> Tuple[bool, Optional[str], VideoDiagnostics | dict[str, object]]:
    """Check video integrity: PyAV reads metadata, Decord samples frames."""
    diagnostics = _empty_video_diagnostics()

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            diagnostics["frame_count"] = stream.frames or 0
            diagnostics["fps"] = float(stream.average_rate) if stream.average_rate else 0.0
            diagnostics["resolution"] = (
                stream.codec_context.width or 0,
                stream.codec_context.height or 0,
            )
            duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
            diagnostics["duration_sec"] = (
                duration
                if duration > 0
                else (diagnostics["frame_count"] / diagnostics["fps"] if diagnostics["fps"] > 0 else 0)
            )

        w, h = diagnostics["resolution"]
        if w <= 0 or h <= 0:
            return False, "无效分辨率", diagnostics

        vr = VideoReader(str(path), ctx=cpu(0))
        total_frames = diagnostics["frame_count"] or len(vr)
        sample_indices = range(0, total_frames, sample_interval)

        prev_frame = None
        consecutive_fail = 0
        timestamps = []

        for idx in sample_indices:
            try:
                frame = vr[idx].asnumpy()
                diagnostics["actual_frames"] += 1
                consecutive_fail = 0

                timestamps.append(_decord_timestamp_ms(vr, idx, diagnostics["fps"]))

                gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
                _accumulate_luminance_flags(gray, diagnostics)

                if prev_frame is not None:
                    _accumulate_bad_frame(float(np.mean(np.abs(gray - prev_frame))), diagnostics)

                prev_frame = gray

            except Exception:
                consecutive_fail += 1
                if consecutive_fail > 5:
                    return False, f"Continuous reads failed at frame {idx}", diagnostics
                continue

        _finalize_frame_diagnostics(timestamps, diagnostics)

        message = _bad_frame_ratio_message(diagnostics, len(sample_indices))
        if message is not None:
            return False, message, diagnostics

        return True, None, diagnostics

    except Exception as e:
        return False, str(e), diagnostics


def check_image_integrity(path: Path) -> Tuple[bool, Optional[str], VideoDiagnostics | dict[str, object]]:
    """Check image integrity."""
    try:
        array = read_image_rgb(path)

        h, w = array.shape[:2]
        if h <= 0 or w <= 0:
            return False, f"Invalid size: {w}x{h}", {}

        gray = np.mean(array, axis=2)
        mean_gray = np.mean(gray)
        if mean_gray < 8.0:
            return False, "Image is almost entirely black", {}
        if mean_gray > 245.0:
            return False, "Image is almost entirely white", {}

        unique_colors = len(np.unique(array))
        if unique_colors < 2:
            return False, f"Too few colors: {unique_colors} kind of color", {}

        return True, None, {}
    except Exception as e:
        return False, str(e), {}


def check_media_integrity(
    path: Path, media_type: DatasetType, sample_interval: int = 30
) -> Tuple[bool, Optional[str], VideoDiagnostics | dict[str, object]]:
    """Unified media integrity check entry point."""
    if media_type == DatasetType.VIDEO:
        return check_video_integrity(path, sample_interval)
    if media_type == DatasetType.IMAGE:
        return check_image_integrity(path)
    return False, f"Unknown media type: {media_type}", {}
