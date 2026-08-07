# deploy/core/preprocessor.py

import cv2
import numpy as np
import torch
from loguru import logger
from pathlib import Path
from typing import List, Union

from .runtime_config import cfg

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


class Preprocessor:
    """Image and video preprocessor."""

    def __init__(self):
        self.cfg = cfg

    def process_image(self, file_path: Path) -> torch.Tensor:
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"Failed to decode: {file_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.cfg.input_size, self.cfg.input_size))
        return torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0)

    def process_image_from_array(self, image_np: np.ndarray) -> torch.Tensor:
        """Process a numpy image array (H, W, 3) RGB uint8."""
        if image_np is None:
            raise ValueError("Image array is None")
        if image_np.shape[-1] != 3 or image_np.dtype != np.uint8:
            raise ValueError(f"Unsupported image format: shape={image_np.shape}, dtype={image_np.dtype}")

        resized = cv2.resize(image_np, (self.cfg.input_size, self.cfg.input_size))
        return torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)

    def process_video(self, file_path: Path) -> torch.Tensor:
        if DECORD_AVAILABLE:
            try:
                return self._process_decord(file_path)
            except Exception as e:
                logger.warning(f"[WARN] Decord failed: {e}")
        return self._process_opencv(file_path)

    def _process_decord(self, file_path: Path) -> torch.Tensor:
        vr = VideoReader(str(file_path), ctx=cpu(0))
        total_frames = len(vr)

        if total_frames >= self.cfg.num_frames:
            indices = list(range(self.cfg.num_frames))
        else:
            indices = list(range(total_frames))

        frames = vr.get_batch(indices).asnumpy()
        if frames.size == 0:
            raise ValueError(f"Empty frames: {file_path}")

        if len(frames) < self.cfg.num_frames:
            logger.warning(f"[WARN] Insufficient frames: {len(frames)}/{self.cfg.num_frames}")

        resized = [cv2.resize(f, (self.cfg.input_size, self.cfg.input_size)) for f in frames]
        video_np = np.stack(resized)
        tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float().div(255.0)

        if tensor.size(0) < self.cfg.num_frames:
            pad = tensor[-1].unsqueeze(0).repeat(self.cfg.num_frames - tensor.size(0), 1, 1, 1)
            tensor = torch.cat([tensor, pad], dim=0)

        return tensor

    def _process_opencv(self, file_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {file_path}")

        frames = []
        while len(frames) < self.cfg.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.cfg.input_size, self.cfg.input_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No frames: {file_path}")

        video_np = np.stack(frames)
        tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float().div(255.0)

        if tensor.size(0) < self.cfg.num_frames:
            pad = tensor[-1].unsqueeze(0).repeat(self.cfg.num_frames - tensor.size(0), 1, 1, 1)
            tensor = torch.cat([tensor, pad], dim=0)

        return tensor
