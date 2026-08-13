from typing import Optional, Sequence

import cv2
import numpy as np
import torch


IMAGENET_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    """Convert one RGB uint8 image [H, W, 3] to normalized [3, input_size, input_size]."""
    if image_np is None:
        raise ValueError("Image array is None")
    if image_np.ndim != 3 or image_np.shape[-1] != 3 or image_np.dtype != np.uint8:
        raise ValueError(f"Unsupported RGB image format: shape={image_np.shape}, dtype={image_np.dtype}")

    resized = cv2.resize(image_np, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div(255.0)
    return normalize_imagenet_tensor(tensor)


def rgb_video_array_to_imagenet_tensor(
    frames_np: np.ndarray,
    input_size: int = IMAGENET_INPUT_SIZE,
) -> torch.Tensor:
    """Convert RGB uint8 video frames [F, H, W, 3] to normalized [F, 3, input_size, input_size]."""
    if frames_np is None:
        raise ValueError("Video frame array is None")
    if frames_np.ndim != 4 or frames_np.shape[-1] != 3 or frames_np.dtype != np.uint8:
        raise ValueError(f"Unsupported RGB video format: shape={frames_np.shape}, dtype={frames_np.dtype}")
    if frames_np.shape[0] == 0:
        raise ValueError("Video frame array is empty")

    resized = [
        cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        for frame in frames_np
    ]
    video_np = np.stack(resized)
    tensor = torch.from_numpy(np.ascontiguousarray(video_np)).permute(0, 3, 1, 2).float().div(255.0)
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
