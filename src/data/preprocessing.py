from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from decord import VideoReader, cpu
from loguru import logger
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from src.data.image_io import read_image_rgb
from src.utils.registry import Registry

IMAGENET_INPUT_SIZE = 224
SWIN_T_RESIZE_SIZE = 232
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingAction:
    """Metadata describing one registered media preprocessing action."""

    key: str
    media_type: str
    method: str
    steps: tuple[str, ...]


PREPROCESSING_REGISTRY = Registry[PreprocessingAction]("preprocessing_actions")
PREPROCESSING_REGISTRY.register(
    "image_imagenet",
    PreprocessingAction(
        key="image_imagenet",
        media_type="image",
        method="rgb_array_to_imagenet_tensor",
        steps=("RGB uint8 validation", "scale to [0, 1]", "bicubic resize", "center crop", "ImageNet normalize"),
    ),
)
PREPROCESSING_REGISTRY.register(
    "video_imagenet",
    PreprocessingAction(
        key="video_imagenet",
        media_type="video",
        method="rgb_video_array_to_imagenet_tensor",
        steps=("RGB frame validation", "scale to [0, 1]", "bicubic resize", "center crop", "ImageNet normalize"),
    ),
)


def preprocessing_actions() -> list[PreprocessingAction]:
    """Return the registered preprocessing action metadata."""
    return [action for _, action in PREPROCESSING_REGISTRY.items()]


def normalize_imagenet_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Normalize RGB tensors that are already scaled to [0, 1].

    Supported shapes:
    - [C, H, W]
    - [F, C, H, W]
    - [B, C, H, W]
    - [B, F, C, H, W]
    """
    if tensor.shape[-3] != len(mean):
        raise ValueError(f"Expected channel dimension at -3 to be {len(mean)}, got shape={tuple(tensor.shape)}")

    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)

    view_shape = [1] * tensor.ndim
    view_shape[-3] = len(mean)
    return (tensor - mean_t.view(view_shape)) / std_t.view(view_shape)


def rgb_array_to_imagenet_tensor(
    image_np: np.ndarray,
    input_size: int = IMAGENET_INPUT_SIZE,
) -> torch.Tensor:
    """Apply the torchvision Swin-T ImageNet resize/crop/normalize pipeline."""
    if image_np is None:
        raise ValueError("Image array is None")
    if image_np.ndim != 3 or image_np.shape[-1] != 3 or image_np.dtype != np.uint8:
        raise ValueError(f"Unsupported RGB image format: shape={image_np.shape}, dtype={image_np.dtype}")

    # ascontiguousarray can return the original read-only Pillow-backed buffer.
    writable_image = (
        np.array(image_np, copy=True, order="C") if not image_np.flags.writeable else np.ascontiguousarray(image_np)
    )
    tensor = torch.from_numpy(writable_image).permute(2, 0, 1).float().div(255.0)
    resize_size = round(input_size * SWIN_T_RESIZE_SIZE / IMAGENET_INPUT_SIZE)
    tensor = tvf.resize(tensor, resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    tensor = tvf.center_crop(tensor, [input_size, input_size])
    return normalize_imagenet_tensor(tensor)


def rgb_video_array_to_imagenet_tensor(
    frames_np: np.ndarray,
    input_size: int = IMAGENET_INPUT_SIZE,
) -> torch.Tensor:
    """Apply the Swin-T ImageNet preprocessing pipeline to RGB video frames."""
    if frames_np is None:
        raise ValueError("Video frame array is None")
    if frames_np.ndim != 4 or frames_np.shape[-1] != 3 or frames_np.dtype != np.uint8:
        raise ValueError(f"Unsupported RGB video format: shape={frames_np.shape}, dtype={frames_np.dtype}")
    if frames_np.shape[0] == 0:
        raise ValueError("Video frame array is empty")

    writable_frames = (
        np.array(frames_np, copy=True, order="C") if not frames_np.flags.writeable else np.ascontiguousarray(frames_np)
    )
    tensor = torch.from_numpy(writable_frames).permute(0, 3, 1, 2).float().div(255.0)
    resize_size = round(input_size * SWIN_T_RESIZE_SIZE / IMAGENET_INPUT_SIZE)
    tensor = tvf.resize(tensor, resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    tensor = tvf.center_crop(tensor, [input_size, input_size])
    return normalize_imagenet_tensor(tensor)


def blank_imagenet_tensor(
    input_size: int = IMAGENET_INPUT_SIZE,
    num_frames: Optional[int] = None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Create a normalized RGB placeholder tensor using [0, 1] pixel fill values."""
    image = torch.full((3, input_size, input_size), fill_value, dtype=torch.float32)
    image = normalize_imagenet_tensor(image)
    if num_frames is None:
        return image
    return image.unsqueeze(0).repeat(num_frames, 1, 1, 1)


class Preprocessor:
    """Image and video preprocessor.

    Images are decoded with Pillow; videos are sampled with Decord.
    """

    def __init__(self, num_frames: int = 8, input_size: int = IMAGENET_INPUT_SIZE):
        if num_frames <= 0 or input_size <= 0:
            raise ValueError("num_frames and input_size must be positive")
        self.num_frames = int(num_frames)
        self.input_size = int(input_size)

    def process_image(self, file_path: Path) -> torch.Tensor:
        """Decode one image and return its ImageNet-style tensor."""
        image_np = read_image_rgb(file_path)
        return rgb_array_to_imagenet_tensor(image_np, input_size=self.input_size)

    def process_image_from_array(self, image_np: np.ndarray) -> torch.Tensor:
        """Process a numpy image array (H, W, 3) RGB uint8."""
        if image_np is None:
            raise ValueError("Image array is None")
        if image_np.ndim != 3 or image_np.shape[-1] != 3 or image_np.dtype != np.uint8:
            raise ValueError(f"Unsupported image format: shape={image_np.shape}, dtype={image_np.dtype}")

        return rgb_array_to_imagenet_tensor(image_np, input_size=self.input_size)

    def process_video(self, file_path: Path) -> torch.Tensor:
        """Sample a fixed number of frames and return their ImageNet-style tensor."""
        try:
            vr = VideoReader(str(file_path), ctx=cpu(0))
        except RuntimeError as exc:
            raise ValueError("Video cannot be decoded") from exc
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError(f"Empty video: {file_path}")

        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        else:
            indices = list(range(total_frames))

        try:
            frames = vr.get_batch(indices).asnumpy()
        except RuntimeError as exc:
            raise ValueError("Video frames cannot be decoded") from exc
        if frames.size == 0:
            raise ValueError(f"Empty frames: {file_path}")

        if len(frames) < self.num_frames:
            logger.warning(f"[WARN] Insufficient frames: {len(frames)}/{self.num_frames}")

        tensor = rgb_video_array_to_imagenet_tensor(frames, input_size=self.input_size)

        if tensor.size(0) < self.num_frames:
            pad = tensor[-1].unsqueeze(0).repeat(self.num_frames - tensor.size(0), 1, 1, 1)
            tensor = torch.cat([tensor, pad], dim=0)

        return tensor


__all__ = [
    "Preprocessor",
    "IMAGENET_INPUT_SIZE",
    "SWIN_T_RESIZE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PreprocessingAction",
    "PREPROCESSING_REGISTRY",
    "preprocessing_actions",
    "normalize_imagenet_tensor",
    "rgb_array_to_imagenet_tensor",
    "rgb_video_array_to_imagenet_tensor",
    "blank_imagenet_tensor",
]
