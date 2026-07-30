# src/data/eda/statistics.py
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from loguru import logger


def analyze_image_properties(
    image_paths: List[Path],
    sample_limit: int = 1000,
    detailed: bool = False,
) -> Dict:
    """Analyze image properties: resolution, aspect ratio, file size."""
    if not image_paths:
        return {"total_files": 0, "error": "No images provided"}

    total = len(image_paths)
    sample_count = min(sample_limit, total) if sample_limit > 0 else total
    sampled = image_paths[:sample_count]

    widths, heights = [], []
    aspect_ratios = []
    file_sizes = []
    color_channels = []

    for img_path in sampled:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            file_sizes.append(img_path.stat().st_size / (1024 * 1024))

            if detailed:
                if len(img.shape) == 2:
                    color_channels.append("grayscale")
                else:
                    c = img.shape[2]
                    if c == 3:
                        color_channels.append("RGB")
                    elif c == 4:
                        color_channels.append("RGBA")
                    else:
                        color_channels.append(f"{c}channels")
        except Exception as e:
            logger.debug(f"Failed to analyze {img_path}: {e}")
            continue

    if not widths:
        return {"total_files": total, "error": "No valid images analyzed"}

    result = {
        "total_files": total,
        "sampled_files": len(widths),
        "width": {
            "min": min(widths),
            "max": max(widths),
            "mean": np.mean(widths),
            "std": np.std(widths),
        },
        "height": {
            "min": min(heights),
            "max": max(heights),
            "mean": np.mean(heights),
            "std": np.std(heights),
        },
        "aspect_ratio": {
            "min": min(aspect_ratios),
            "max": max(aspect_ratios),
            "mean": np.mean(aspect_ratios),
            "std": np.std(aspect_ratios),
        },
        "file_size_mb": {
            "min": min(file_sizes),
            "max": max(file_sizes),
            "mean": np.mean(file_sizes),
            "std": np.std(file_sizes),
        },
    }

    if detailed and color_channels:
        result["color_spaces"] = dict(Counter(color_channels))

    return result


def analyze_video_properties(
    video_paths: List[Path],
    sample_limit: int = 50,
    detailed: bool = False,
) -> Dict:
    """Analyze video properties: FPS, frame count, resolution, duration, codec."""
    if not video_paths:
        return {"total_files": 0, "error": "No videos provided"}

    total = len(video_paths)
    sample_count = min(sample_limit, total) if sample_limit > 0 else total
    sampled = video_paths[:sample_count]

    fps_list = []
    frame_count_list = []
    widths, heights = [], []
    durations = []
    codecs = []

    for video_path in sampled:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps_list.append(fps)
            frame_count_list.append(frame_count)
            widths.append(width)
            heights.append(height)
            durations.append(frame_count / fps if fps > 0 else 0)

            if detailed:
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec_char = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                codecs.append(codec_char)

            cap.release()
        except Exception as e:
            logger.debug(f"Failed to analyze {video_path}: {e}")
            continue

    if not fps_list:
        return {"total_files": total, "error": "No valid videos analyzed"}

    result = {
        "total_files": total,
        "sampled_files": len(fps_list),
        "fps": {
            "min": min(fps_list),
            "max": max(fps_list),
            "mean": np.mean(fps_list),
            "std": np.std(fps_list),
        },
        "frame_count": {
            "min": min(frame_count_list),
            "max": max(frame_count_list),
            "mean": np.mean(frame_count_list),
            "std": np.std(frame_count_list),
        },
        "resolution": {
            "width": {
                "min": min(widths),
                "max": max(widths),
                "mean": np.mean(widths),
                "std": np.std(widths),
            },
            "height": {
                "min": min(heights),
                "max": max(heights),
                "mean": np.mean(heights),
                "std": np.std(heights),
            },
        },
        "duration_sec": {
            "min": min(durations),
            "max": max(durations),
            "mean": np.mean(durations),
            "std": np.std(durations),
        },
    }

    if detailed and codecs:
        result["codecs"] = dict(Counter(codecs))

    return result


def compute_mos_statistics(df: pd.DataFrame, score_col: str = "mos") -> Dict:
    """Compute MOS statistics: distribution, outliers, skewness, kurtosis."""
    if df is None or df.empty or score_col not in df.columns:
        return {"error": f"Column '{score_col}' not found or DataFrame is empty"}

    scores = df[score_col].dropna()
    if scores.empty:
        return {"error": "No valid scores found"}

    mean = scores.mean()
    std = scores.std()
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = scores[(scores < lower_bound) | (scores > upper_bound)]

    return {
        "total_samples": len(scores),
        "range": (float(scores.min()), float(scores.max())),
        "mean": float(mean),
        "std": float(std),
        "median": float(scores.median()),
        "q1": float(q1),
        "q3": float(q3),
        "skew": float(scores.skew()),
        "kurtosis": float(scores.kurtosis()),
        "outlier_count": len(outliers),
        "outlier_ratio": len(outliers) / len(scores) if len(scores) > 0 else 0,
    }


def generate_full_statistics_report(
    image_paths: Optional[List[Path]] = None,
    video_paths: Optional[List[Path]] = None,
    df: Optional[pd.DataFrame] = None,
    score_col: str = "mos",
) -> Dict:
    """Generate a complete dataset statistics report."""
    report = {}

    if image_paths:
        report["images"] = analyze_image_properties(image_paths, detailed=True)

    if video_paths:
        report["videos"] = analyze_video_properties(video_paths, detailed=True)

    if df is not None and not df.empty:
        report["scores"] = compute_mos_statistics(df, score_col)

    report["overview"] = {
        "total_images": len(image_paths) if image_paths else 0,
        "total_videos": len(video_paths) if video_paths else 0,
        "total_samples": len(df) if df is not None else 0,
    }

    return report