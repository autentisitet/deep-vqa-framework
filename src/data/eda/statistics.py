# src/data/eda/statistics.py

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger


def analyze_image_properties(
    image_paths: List[Path], 
    sample_limit: int = 1000,
    detailed: bool = False
) -> Dict:
    """
    分析图像属性

    Args:
        image_paths: 图像文件路径列表
        sample_limit: 采样数量限制（-1 表示全部）
        detailed: 是否输出详细统计

    Returns:
        包含以下字段的字典:
        - total_files: 总文件数
        - sampled_files: 实际采样数
        - resolutions: 分辨率统计 (min, max, mean)
        - aspect_ratios: 宽高比统计
        - color_spaces: 颜色空间分布 (如果 detailed=True)
        - sizes_mb: 文件大小统计 (如果 detailed=True)
    """
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
    detailed: bool = False
) -> Dict:
    """
    分析视频属性

    Args:
        video_paths: 视频文件路径列表
        sample_limit: 采样数量限制
        detailed: 是否输出详细统计

    Returns:
        包含以下字段的字典:
        - total_files: 总文件数
        - sampled_files: 实际采样数
        - fps: 帧率统计 (min, max, mean, std)
        - frame_count: 帧数统计
        - resolution: 分辨率统计
        - duration_sec: 时长统计
        - codec_info: 编码信息 (如果 detailed=True)
    """
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
                # 尝试获取编码信息
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
    """
    计算 MOS/DMOS 统计信息

    Args:
        df: 包含评分列的数据框
        score_col: 评分列名称（默认 "mos"）

    Returns:
        包含以下字段的字典:
        - total_samples: 总样本数
        - mos_range: (min, max)
        - mos_mean: 均值
        - mos_std: 标准差
        - mos_median: 中位数
        - mos_q1: 第一四分位数
        - mos_q3: 第三四分位数
        - mos_skew: 偏度
        - mos_kurtosis: 峰度
        - outlier_count: 离群值数量（3σ）
    """
    if df is None or df.empty or score_col not in df.columns:
        return {"error": f"Column '{score_col}' not found or DataFrame is empty"}

    scores = df[score_col].dropna()

    if scores.empty:
        return {"error": "No valid scores found"}

    mean = scores.mean()
    std = scores.std()
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)

    # 3σ 离群值检测
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = scores[(scores < lower_bound) | (scores > upper_bound)]

    return {
        "total_samples": len(scores),
        "mos_range": (float(scores.min()), float(scores.max())),
        "mos_mean": float(mean),
        "mos_std": float(std),
        "mos_median": float(scores.median()),
        "mos_q1": float(q1),
        "mos_q3": float(q3),
        "mos_skew": float(scores.skew()),
        "mos_kurtosis": float(scores.kurtosis()),
        "outlier_count": len(outliers),
        "outlier_ratio": len(outliers) / len(scores) if len(scores) > 0 else 0,
    }


def generate_full_statistics_report(
    image_paths: List[Path] = None,
    video_paths: List[Path] = None,
    df: pd.DataFrame = None,
    score_col: str = "mos",
) -> Dict:
    """
    生成完整的数据集统计报告

    Args:
        image_paths: 图像文件路径列表
        video_paths: 视频文件路径列表
        df: 包含评分的数据框
        score_col: 评分列名称

    Returns:
        完整的统计报告字典
    """
    report = {}

    if image_paths:
        report["images"] = analyze_image_properties(image_paths, detailed=True)

    if video_paths:
        report["videos"] = analyze_video_properties(video_paths, detailed=True)

    if df is not None and not df.empty:
        report["scores"] = compute_mos_statistics(df, score_col)

    # 数据集概览
    report["overview"] = {
        "total_images": len(image_paths) if image_paths else 0,
        "total_videos": len(video_paths) if video_paths else 0,
        "total_samples": len(df) if df is not None else 0,
    }

    return report


def print_statistics_report(report: Dict):
    """打印统计报告（人类可读格式）"""
    logger.info("=" * 60)
    logger.info("📊 Dataset Statistics Report")
    logger.info("=" * 60)

    overview = report.get("overview", {})
    if overview:
        logger.info(f"  Total Images: {overview.get('total_images', 0)}")
        logger.info(f"  Total Videos: {overview.get('total_videos', 0)}")
        logger.info(f"  Total Samples: {overview.get('total_samples', 0)}")

    images = report.get("images", {})
    if images and "error" not in images:
        logger.info("")
        logger.info("  📷 Image Properties:")
        logger.info(f"    Files: {images.get('total_files', 0)} (sampled: {images.get('sampled_files', 0)})")
        res = images.get("resolution", {})
        if res:
            logger.info(f"    Resolution: {res.get('min', 0)}x{res.get('min_h', 0)} ~ {res.get('max', 0)}x{res.get('max_h', 0)}")
        size = images.get("file_size_mb", {})
        if size:
            logger.info(f"    File Size: {size.get('min', 0):.2f}MB ~ {size.get('max', 0):.2f}MB (mean: {size.get('mean', 0):.2f}MB)")

    videos = report.get("videos", {})
    if videos and "error" not in videos:
        logger.info("")
        logger.info("  🎬 Video Properties:")
        logger.info(f"    Files: {videos.get('total_files', 0)} (sampled: {videos.get('sampled_files', 0)})")
        fps = videos.get("fps", {})
        if fps:
            logger.info(f"    FPS: {fps.get('min', 0):.1f} ~ {fps.get('max', 0):.1f} (mean: {fps.get('mean', 0):.1f})")
        duration = videos.get("duration_sec", {})
        if duration:
            logger.info(f"    Duration: {duration.get('min', 0):.1f}s ~ {duration.get('max', 0):.1f}s (mean: {duration.get('mean', 0):.1f}s)")

    scores = report.get("scores", {})
    if scores and "error" not in scores:
        logger.info("")
        logger.info("  📈 MOS Distribution:")
        logger.info(f"    Samples: {scores.get('total_samples', 0)}")
        logger.info(f"    Range: [{scores.get('mos_range', (0, 0))[0]:.3f}, {scores.get('mos_range', (0, 0))[1]:.3f}]")
        logger.info(f"    Mean: {scores.get('mos_mean', 0):.3f} | Std: {scores.get('mos_std', 0):.3f}")
        logger.info(f"    Skew: {scores.get('mos_skew', 0):.3f} | Kurtosis: {scores.get('mos_kurtosis', 0):.3f}")
        logger.info(f"    Outliers (3σ): {scores.get('outlier_count', 0)} ({scores.get('outlier_ratio', 0)*100:.1f}%)")

    logger.info("=" * 60)