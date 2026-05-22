# src/data/eda/integrity.py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from loguru import logger
from src.data.types import DatasetType

# TODO:
# 检查是否有sample data损坏、缺失、重复
# 检查file name是否和label匹配: assert set(img_names) == set(labels['image_id'])


# 尝试导入 Decord，失败时回退到 OpenCV
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
    logger.info("✅ 使用 Decord 进行视频处理（高性能模式）")
except ImportError:
    import cv2
    DECORD_AVAILABLE = False
    logger.warning("⚠️ Decord 未安装，回退到 OpenCV（性能较低）")


def check_video_integrity_decord(path: Path, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    使用 Decord 检查视频完整性
    优势：
    - 静态链接 FFmpeg，无系统依赖问题
    - 10倍于 OpenCV 的性能
    - 支持随机帧访问
    """
    diagnostics = {
        "frame_count": 0,
        "bad_frames": 0,
        "black_frames": 0,
        "actual_frames": 0,
        "fps": 0,
        "resolution": (0, 0),
        "duration_sec": 0
    }

    try:
        # Decord 打开视频
        vr = VideoReader(str(path), ctx=cpu(0))

        # 基本信息
        diagnostics["frame_count"] = len(vr)
        diagnostics["fps"] = vr.get_avg_fps()
        diagnostics["resolution"] = (vr[0].shape[1], vr[0].shape[0])  # width, height
        diagnostics["duration_sec"] = diagnostics["frame_count"] / diagnostics["fps"] if diagnostics["fps"] > 0 else 0

        # 分辨率检查
        w, h = diagnostics["resolution"]
        if w <= 0 or h <= 0:
            return False, "无效分辨率", diagnostics

        # 采样检查帧完整性（Decord 随机访问，效率高）
        total_frames = diagnostics["frame_count"]
        sample_indices = range(0, total_frames, sample_interval)

        prev_frame = None
        consecutive_fail = 0

        for idx in sample_indices:
            try:
                # Decord 直接随机访问指定帧
                frame = vr[idx].asnumpy()
                diagnostics["actual_frames"] += 1
                consecutive_fail = 0

                # 检查黑帧
                gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
                if np.mean(gray) < 8.0:
                    diagnostics["black_frames"] += 1

                # 检查卡帧（与前帧对比）
                if prev_frame is not None:
                    diff = np.mean(np.abs(gray - prev_frame))
                    if diff < 0.5:
                        diagnostics["bad_frames"] += 1

                prev_frame = gray

            except Exception as e:
                consecutive_fail += 1
                if consecutive_fail > 5:
                    return False, f"在帧 {idx} 处连续读取失败", diagnostics
                continue

        # 检查坏帧比例
        sample_count = max(1, len(sample_indices))
        bad_ratio = diagnostics["bad_frames"] / sample_count
        if bad_ratio > 0.3:
            return False, f"坏帧比例过高: {bad_ratio:.2%}", diagnostics

        return True, None, diagnostics

    except Exception as e:
        return False, str(e), diagnostics


def check_video_integrity_fallback(path: Path, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    回退方案：使用 OpenCV 检查视频完整性（保持原有逻辑）
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "无法打开视频", {}

    diagnostics = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "bad_frames": 0,
        "black_frames": 0,
        "actual_frames": 0,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "resolution": (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ),
        "duration_sec": 0
    }

    if diagnostics["fps"] > 0:
        diagnostics["duration_sec"] = diagnostics["frame_count"] / diagnostics["fps"]

    w, h = diagnostics["resolution"]
    if w <= 0 or h <= 0:
        cap.release()
        return False, "无效分辨率", diagnostics

    frame_idx = 0
    prev_gray = None
    consecutive_fail = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_fail += 1
            if consecutive_fail > 5:
                break
            continue

        consecutive_fail = 0
        diagnostics["actual_frames"] += 1

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 8.0:
                diagnostics["black_frames"] += 1

            if prev_gray is not None:
                diff = np.mean(cv2.absdiff(gray, prev_gray))
                if diff < 0.5:
                    diagnostics["bad_frames"] += 1

            prev_gray = gray

        frame_idx += 1
        if frame_idx >= diagnostics["frame_count"]:
            break

    cap.release()

    # 检查帧数匹配
    if abs(diagnostics["actual_frames"] - diagnostics["frame_count"]) > diagnostics["frame_count"] * 0.1:
        return False, f"帧数不匹配", diagnostics

    sample_count = max(1, diagnostics["frame_count"] // sample_interval)
    bad_ratio = diagnostics["bad_frames"] / sample_count
    if bad_ratio > 0.3:
        return False, f"坏帧比例过高: {bad_ratio:.2%}", diagnostics

    return True, None, diagnostics


def check_video_integrity(path: Path, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    统一的视频完整性检查入口
    自动选择最优后端（Decord > OpenCV）
    """
    if DECORD_AVAILABLE:
        return check_video_integrity_decord(path, sample_interval)
    else:
        logger.debug("Decord 不可用，使用 OpenCV 回退方案")
        return check_video_integrity_fallback(path, sample_interval)


def check_image_integrity(path: Path) -> Tuple[bool, Optional[str], Dict]:
    """检查图像完整性（保持不变）"""
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return False, "无法解码"

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return False, f"无效尺寸: {w}x{h}", {}

        unique_colors = len(np.unique(img))
        if unique_colors < 2:
            return False, f"颜色过少: {unique_colors}种颜色", {}

        return True, None, {}
    except Exception as e:
        return False, str(e), {}


def check_media_integrity(path: Path, media_type: DatasetType, sample_interval: int = 30) -> Tuple[bool, Optional[str], Dict]:
    """
    统一的媒体完整性检查入口

    Args:
        path: 文件路径
        media_type: 媒体类型（IMAGE 或 VIDEO）
        sample_interval: 视频采样间隔

    Returns:
        (是否完整, 错误信息, 诊断信息)
    """
    if media_type == DatasetType.VIDEO:
        is_ok, error, diagnostics = check_video_integrity(path, sample_interval)
        return is_ok, error, diagnostics
    elif media_type == DatasetType.IMAGE:
        is_ok, error, diagnostics = check_image_integrity(path)
        return is_ok, error, diagnostics
    else:
        return False, f"未知媒体类型: {media_type}", {}