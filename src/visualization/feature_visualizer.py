"""Feature-map and Grad-CAM visualization for IQA/VQA models.

The utilities deliberately operate through hooks, so the model's production
forward path remains unchanged.  They support both NCHW (ResNet) and NHWC
(Torchvision Swin) feature layouts used by :class:`IQAVQANet`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class FeatureVisualizationResult:
    score: float
    activations: dict[str, torch.Tensor]
    cam: torch.Tensor | None = None


def to_spatial_feature_map(value: torch.Tensor) -> torch.Tensor:
    """Convert a feature tensor to [B,C,H,W] when it is spatial."""
    if value.ndim == 3:
        batch, tokens, channels = value.shape
        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError(f"Expected square spatial tokens, got {tuple(value.shape)}")
        return value.permute(0, 2, 1).contiguous().view(batch, channels, side, side)
    if value.ndim != 4:
        raise ValueError(f"Expected a 4D spatial activation, got {tuple(value.shape)}")
    # Swin emits [B,H,W,C], while ResNet emits [B,C,H,W]. In both cases the
    # spatial dimensions are usually small relative to the channel count.
    if value.shape[-1] > value.shape[1] and value.shape[-1] > value.shape[2]:
        return value.permute(0, 3, 1, 2).contiguous()
    return value


def _normalise_map(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    value = value - value.amin(dim=(-2, -1), keepdim=True)
    return value / value.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class FeatureVisualizer:
    """Capture intermediate feature maps and generate regression-target Grad-CAM."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu"):
        self.model = model
        self.device = torch.device(device)

    def _resolve_module(self, name: str) -> nn.Module:
        modules = dict(self.model.named_modules())
        if name not in modules:
            choices = [key for key in modules if key.startswith("image_backbone")]
            raise KeyError(f"Unknown layer '{name}'. Example available image layers: {choices[-12:]}")
        return modules[name]

    def image_layer_names(self) -> list[str]:
        """Return stable spatial targets for image feature visualization."""
        # The complete backbone output is spatial for both supported families:
        # Swin emits NHWC and ResNet emits NCHW. Nested Swin MLP modules emit
        # token vectors and are intentionally excluded from the default list.
        if "image_backbone" not in dict(self.model.named_modules()):
            return []
        return ["image_backbone"]

    def default_image_layers(self) -> list[str]:
        """Choose the last few spatial stages for a standard model overview."""
        names = self.image_layer_names()
        if not names:
            raise RuntimeError("The loaded model does not expose spatial image backbone layers")
        return names[-3:]

    @torch.enable_grad()
    def run_image(
        self,
        image: torch.Tensor,
        layers: Iterable[str] = (),
        cam_layer: str | None = None,
    ) -> FeatureVisualizationResult:
        """Run one image batch and capture activations plus optional Grad-CAM."""
        # Required when the backbone is frozen: otherwise its output may not
        # carry a grad function and Grad-CAM would silently have no gradient.
        image = image.to(self.device).detach().requires_grad_(bool(cam_layer))
        captured: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}
        hooks = []
        requested = list(dict.fromkeys(layers))
        if cam_layer and cam_layer not in requested:
            requested.append(cam_layer)

        for name in requested:
            module = self._resolve_module(name)

            def save_activation(_module, _inputs, output, key=name):
                if isinstance(output, torch.Tensor):
                    captured[key] = output
                    if output.requires_grad:
                        output.register_hook(lambda grad, k=key: gradients.__setitem__(k, grad))

            hooks.append(module.register_forward_hook(save_activation))

        try:
            self.model.zero_grad(set_to_none=True)
            score = self.model(image).reshape(-1)[0]
            cam = None
            if cam_layer:
                if cam_layer not in captured or cam_layer not in gradients:
                    raise ValueError(f"Layer '{cam_layer}' did not produce a differentiable spatial activation")
                activation = to_spatial_feature_map(captured[cam_layer])[0:1]
                gradient = to_spatial_feature_map(gradients[cam_layer])[0:1]
                weights = gradient.mean(dim=(-2, -1), keepdim=True)
                cam = _normalise_map(torch.relu((weights * activation).sum(dim=1, keepdim=True)))[:, 0]
        finally:
            for hook in hooks:
                hook.remove()
        return FeatureVisualizationResult(float(score.detach().cpu()), captured, cam)

    @staticmethod
    def save_feature_grid(activation: torch.Tensor, path: Path, max_channels: int = 16) -> None:
        """Save mean-activation and up to ``max_channels`` individual maps."""
        maps = _normalise_map(to_spatial_feature_map(activation)[0].abs())
        count = min(max_channels, maps.shape[0])
        columns = min(4, count + 1)
        rows = int(np.ceil((count + 1) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
        panels = [maps.mean(dim=0), *maps[:count]]
        titles = ["mean |activation|", *[f"channel {i}" for i in range(count)]]
        for axis, panel, title in zip(axes.flat, panels, titles):
            axis.imshow(panel.cpu(), cmap="magma", vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")
        for axis in axes.flat[len(panels) :]:
            axis.axis("off")
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def save_cam_overlay(image: torch.Tensor, cam: torch.Tensor, path: Path, alpha: float = 0.45) -> None:
        """Save a Grad-CAM overlay; image and CAM are expected in model crop space."""
        image = image[0].detach().cpu().float()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        rgb = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        heat = cam[0].detach().cpu().numpy()
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(rgb)
        axis.imshow(heat, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axis.axis("off")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close(figure)


def load_image_tensor(path: Path, input_size: int) -> torch.Tensor:
    """Load an image using the same preprocessing as deployment."""
    from src.data.preprocessing import rgb_array_to_imagenet_tensor

    with Image.open(path) as image:
        return rgb_array_to_imagenet_tensor(np.asarray(image.convert("RGB")), input_size).unsqueeze(0)


__all__ = ["FeatureVisualizationResult", "FeatureVisualizer", "load_image_tensor", "to_spatial_feature_map"]
