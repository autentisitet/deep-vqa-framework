# src/data/dataset_types.py
from enum import Enum
from typing import Set, Union

# 放在类外部，避免被 Enum 捕获
_VIDEO_EXTS: Set[str] = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".wmv", ".flv", ".3gp", ".m4v", ".ts",
    ".mpeg", ".mpg", ".m2ts", ".mxf",
}

_IMAGE_EXTS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    ".gif", ".ico", ".heic", ".heif", ".avif",
    ".ppm", ".pgm", ".pbm", ".pnm",
}


class DatasetType(Enum):
    IMAGE = "image"
    VIDEO = "video"

    @classmethod
    def detect(cls, ext: str) -> "DatasetType":
        ext = ext.lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}"

        if ext in _VIDEO_EXTS:
            return cls.VIDEO
        elif ext in _IMAGE_EXTS:
            return cls.IMAGE
        raise ValueError(f"Unknown extension: {ext}")

    @classmethod
    def parse(cls, value: str) -> "DatasetType":
        value = value.lower().strip()
        if value == "video":
            return cls.VIDEO
        elif value == "image":
            return cls.IMAGE
        raise ValueError(f"Unknown dataset type: {value}")

    @classmethod
    def all_extensions(cls) -> Set[str]:
        return _VIDEO_EXTS | _IMAGE_EXTS

    @classmethod
    def extensions_for(cls, dataset_type: Union[str, "DatasetType"]) -> Set[str]:
        if isinstance(dataset_type, str):
            dataset_type = cls.parse(dataset_type)
        if dataset_type == cls.VIDEO:
            return _VIDEO_EXTS
        return _IMAGE_EXTS

    @classmethod
    def is_video(cls, ext: str) -> bool:
        try:
            return cls.detect(ext) == cls.VIDEO
        except ValueError:
            return False

    @classmethod
    def is_image(cls, ext: str) -> bool:
        try:
            return cls.detect(ext) == cls.IMAGE
        except ValueError:
            return False