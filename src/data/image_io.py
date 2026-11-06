"""Image decoding helpers.

Pillow owns image I/O here: it covers more formats than OpenCV and reads RGB
natively. These helpers additionally apply EXIF orientation, which
``cv2.imread`` silently ignores.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def read_image_rgb(path: Path) -> np.ndarray:
    """Decode an image as an RGB uint8 array with EXIF orientation applied."""
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            return np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Image cannot be decoded") from exc


def image_size_oriented(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` without decoding pixels, honoring EXIF."""
    with Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(0x0112, 1)
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        return width, height
