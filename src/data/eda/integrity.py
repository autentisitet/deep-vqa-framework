# src/data/eda/integrity.py
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from src.data.types import DatasetType

# TODO:
# 检查是否有sample data损坏、缺失、重复
# 检查file name是否和label匹配: assert set(img_names) == set(labels['image_id'])


# Attempt to import Decord, fall back to OpenCV if it fails
try:
    from decord import VideoReader, cpu

    DECORD_AVAILABLE = True
    logger.info("✅ Video processing using Decord (high-performance mode)")
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("⚠️ Decord is not installed; revert to OpenCV (lower performance).")


def check_video_integrity_decord(path: Path, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    Using Decord to check video integrity

    Advantages:

    - Static linking to FFmpeg, no system dependency issues

    - 10 times faster than OpenCV

    - Supports random frame access
    """
    diagnostics = {
        "frame_count": 0,
        "bad_frames": 0,
        "black_frames": 0,
        "white_frames": 0,           # ✅ 新增：白帧计数
        "actual_frames": 0,
        "fps": 0,
        "resolution": (0, 0),
        "duration_sec": 0,
        "frame_drops": False,        # ✅ 新增：是否有跳帧
        "irregular_interval_ratio": 0.0,  # ✅ 新增：帧间隔异常比例
    }

    try:
        # Decord Open video
        vr = VideoReader(str(path), ctx=cpu(0))

        # Basic Information
        diagnostics["frame_count"] = len(vr)
        diagnostics["fps"] = vr.get_avg_fps()
        diagnostics["resolution"] = (vr[0].shape[1], vr[0].shape[0])  # width, height
        diagnostics["duration_sec"] = diagnostics["frame_count"] / diagnostics["fps"] if diagnostics["fps"] > 0 else 0

        # Resolution check
        w, h = diagnostics["resolution"]
        if w <= 0 or h <= 0:
            return False, "无效分辨率", diagnostics

        # Sample to check frame integrity (Decord random access, high efficiency)
        total_frames = diagnostics["frame_count"]
        sample_indices = range(0, total_frames, sample_interval)

        prev_frame = None
        consecutive_fail = 0
        timestamps = []              # ✅ 新增：记录时间戳用于跳帧检测

        for idx in sample_indices:
            try:
                # Decord directly accesses a specified frame randomly.
                frame = vr[idx].asnumpy()
                diagnostics["actual_frames"] += 1
                consecutive_fail = 0

                # ✅ 获取时间戳（毫秒）
                try:
                    timestamp = vr.get_frame_timestamp(idx)[0] * 1000  # 秒转毫秒
                    timestamps.append(timestamp)
                except:
                    # 如果 decord 不支持 get_frame_timestamp，用帧索引估算
                    timestamps.append(idx / diagnostics["fps"] * 1000)

                # 转灰度图
                gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
                mean_gray = np.mean(gray)

                # ✅ 检查黑帧
                if mean_gray < 8.0:
                    diagnostics["black_frames"] += 1

                # ✅ 检查白帧（新增）
                if mean_gray > 245.0:
                    diagnostics["white_frames"] += 1

                # 检查帧间差异（坏帧/卡顿检测）
                if prev_frame is not None:
                    diff = np.mean(np.abs(gray - prev_frame))
                    if diff < 0.5:
                        diagnostics["bad_frames"] += 1

                prev_frame = gray

            except Exception:
                consecutive_fail += 1
                if consecutive_fail > 5:
                    return False, f"Continuous reads failed at frame {idx}", diagnostics
                continue

        # ✅ 跳帧检测：检查帧间隔是否均匀
        if len(timestamps) > 2:
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # 如果标准差 > 均值的 20%，说明帧间隔不均匀（可能有跳帧）
            irregular_ratio = std_interval / (mean_interval + 1e-6)
            diagnostics["irregular_interval_ratio"] = irregular_ratio
            
            # 如果有某个间隔超过平均间隔的 1.5 倍，说明可能有跳帧
            max_interval = np.max(intervals)
            if max_interval > mean_interval * 1.5:
                diagnostics["frame_drops"] = True

        # Check the percentage of bad frames
        sample_count = max(1, len(sample_indices))
        bad_ratio = diagnostics["bad_frames"] / sample_count
        if bad_ratio > 0.3:
            return False, f"Too high percentage of bad frames: {bad_ratio:.2%}", diagnostics

        return True, None, diagnostics

    except Exception as e:
        return False, str(e), diagnostics


def check_video_integrity_fallback(path: Path, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    Rollback solution: Use OpenCV to check video integrity (maintain original logic).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "Unable to open video", {}

    diagnostics = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "bad_frames": 0,
        "black_frames": 0,
        "white_frames": 0,           # ✅ 新增：白帧计数
        "actual_frames": 0,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "resolution": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "duration_sec": 0,
        "frame_drops": False,        # ✅ 新增：是否有跳帧
        "irregular_interval_ratio": 0.0,  # ✅ 新增：帧间隔异常比例
    }

    if diagnostics["fps"] > 0:
        diagnostics["duration_sec"] = diagnostics["frame_count"] / diagnostics["fps"]

    w, h = diagnostics["resolution"]
    if w <= 0 or h <= 0:
        cap.release()
        return False, "Invalid resolution", diagnostics

    frame_idx = 0
    prev_gray = None
    consecutive_fail = 0
    timestamps = []                  # ✅ 新增：记录时间戳
    prev_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_fail += 1
            if consecutive_fail > 5:
                break
            continue

        consecutive_fail = 0
        diagnostics["actual_frames"] += 1

        # ✅ 获取时间戳
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp > 0:
            timestamps.append(timestamp)

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_gray = np.mean(gray)

            # ✅ 检查黑帧
            if mean_gray < 8.0:
                diagnostics["black_frames"] += 1

            # ✅ 检查白帧（新增）
            if mean_gray > 245.0:
                diagnostics["white_frames"] += 1

            if prev_gray is not None:
                diff = np.mean(cv2.absdiff(gray, prev_gray))
                if diff < 0.5:
                    diagnostics["bad_frames"] += 1

            prev_gray = gray

        frame_idx += 1
        if frame_idx >= diagnostics["frame_count"]:
            break

    cap.release()

    # ✅ 跳帧检测：检查帧间隔是否均匀
    if len(timestamps) > 2:
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        irregular_ratio = std_interval / (mean_interval + 1e-6)
        diagnostics["irregular_interval_ratio"] = irregular_ratio
        
        max_interval = np.max(intervals)
        if max_interval > mean_interval * 1.5:
            diagnostics["frame_drops"] = True

    # Check frame count matching
    if abs(diagnostics["actual_frames"] - diagnostics["frame_count"]) > diagnostics["frame_count"] * 0.1:
        return False, "Frame rate mismatch", diagnostics

    sample_count = max(1, diagnostics["frame_count"] // sample_interval)
    bad_ratio = diagnostics["bad_frames"] / sample_count
    if bad_ratio > 0.3:
        return False, f"Too high percentage of bad frames: {bad_ratio:.2%}", diagnostics

    return True, None, diagnostics


def check_video_integrity(path: Path, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    A unified entry point for video integrity checks
    Automatically selects the optimal backend (Decord > OpenCV)
    """
    if DECORD_AVAILABLE:
        return check_video_integrity_decord(path, sample_interval)
    else:
        logger.debug("Decord is unavailable; use the OpenCV fallback solution.")
        return check_video_integrity_fallback(path, sample_interval)


def check_image_integrity(path: Path) -> Tuple[bool, Optional[str], Dict]:
    """Check image integrity"""
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return False, "Unable to decode"

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return False, f"Invalid size: {w}x{h}", {}

        # ✅ 检查是否为全黑/全白图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        if mean_gray < 8.0:
            return False, "Image is almost entirely black", {}
        if mean_gray > 245.0:
            return False, "Image is almost entirely white", {}

        unique_colors = len(np.unique(img))
        if unique_colors < 2:
            return False, f"Too few colors: {unique_colors} kind of color", {}

        return True, None, {}
    except Exception as e:
        return False, str(e), {}


def check_media_integrity(path: Path, media_type: DatasetType, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    Unified Media Integrity Check Entry Point

    ## Args:

    - path: File path

    - media_type: Media type (IMAGE or VIDEO)

    - sample_interval: Video sampling interval

    ## Returns:

    - (Integrity status, error message, diagnostic information)
    """
    if media_type == DatasetType.VIDEO:
        is_ok, error, diagnostics = check_video_integrity(path, sample_interval)
        return is_ok, error, diagnostics
    elif media_type == DatasetType.IMAGE:
        is_ok, error, diagnostics = check_image_integrity(path)
        return is_ok, error, diagnostics
    else:
        return False, f"Unknown media type: {media_type}", {}