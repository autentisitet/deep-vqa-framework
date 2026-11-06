# src/utils/file_loader.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

from loguru import logger

from src.data.dataset_types import DatasetType


@dataclass
class AssetInfo:
    path: Path
    filename: str
    extension: str

    @property
    def dataset_type(self) -> DatasetType:
        """Classify this asset by its file extension."""
        return DatasetType.detect(self.extension)

    @property
    def is_video(self) -> bool:
        """Whether this asset is a video."""
        return self.dataset_type == DatasetType.VIDEO

    @property
    def is_image(self) -> bool:
        """Whether this asset is an image."""
        return self.dataset_type == DatasetType.IMAGE


class CaseInsensitiveAssetResolver:
    def __init__(self, target_dir: Union[str, Path]):
        self.target_dir = Path(target_dir).resolve()
        self.full_registry: Dict[str, AssetInfo] = {}

        logger.info(f"Scanning: {self.target_dir}")

        all_exts = DatasetType.all_extensions()
        for f in filter(Path.is_file, self.target_dir.rglob("*")):
            ext = f.suffix.lower()
            if ext in all_exts:
                self.full_registry[f.name.lower()] = AssetInfo(
                    path=f,
                    filename=f.name,
                    extension=ext,
                )

        logger.info(f"Indexed {len(self.full_registry)} files")

    def find_asset(self, filename: str) -> AssetInfo:
        """Return the indexed asset for a filename, matching case-insensitively."""
        key = Path(filename).name.lower()
        if key in self.full_registry:
            return self.full_registry[key]
        raise FileNotFoundError(f"File not found: {filename} in {self.target_dir}")

    def find_path(self, filename: str) -> Path:
        """Return the filesystem path of a resolved asset."""
        return self.find_asset(filename).path

    def find_with_info(self, filename: str) -> Tuple[Path, bool, bool]:
        """Return ``(path, is_video, is_image)`` for a resolved asset."""
        asset = self.find_asset(filename)
        return asset.path, asset.is_video, asset.is_image

    def contains(self, filename: str) -> bool:
        """Whether a filename has been indexed."""
        return Path(filename).name.lower() in self.full_registry
