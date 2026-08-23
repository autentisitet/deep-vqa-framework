import numpy as np

from src.data.dataset_loaders import MetadataLoaderFactory
from src.data.dataset_types import DatasetType
from src.data.preprocessing import (
    PREPROCESSING_REGISTRY,
    rgb_array_to_imagenet_tensor,
    rgb_video_array_to_imagenet_tensor,
)


def test_dataset_registry_contains_supported_metadata_loaders() -> None:
    assert set(MetadataLoaderFactory.available_datasets()) == {"konvid-1k", "t2vqa-db", "tid2013"}
    assert MetadataLoaderFactory.normalize_key(" TID2013 ") == "tid2013"


def test_dataset_type_detection_is_case_insensitive() -> None:
    assert DatasetType.detect("JPG") is DatasetType.IMAGE
    assert DatasetType.detect(".M2TS") is DatasetType.VIDEO


def test_media_extensions_have_stable_and_experimental_contracts() -> None:
    assert ".jpg" in DatasetType.stable_extensions()
    assert ".mp4" in DatasetType.stable_extensions()
    assert ".heic" in DatasetType.experimental_extensions()
    assert DatasetType.is_experimental(".HEIC")
    assert ".heic" not in DatasetType.stable_extensions()


def test_registered_preprocessing_pipelines_match_tensor_shapes() -> None:
    assert set(PREPROCESSING_REGISTRY.keys()) == {"image_imagenet", "video_imagenet"}
    image = rgb_array_to_imagenet_tensor(np.zeros((32, 48, 3), dtype=np.uint8), input_size=32)
    video = rgb_video_array_to_imagenet_tensor(np.zeros((2, 32, 48, 3), dtype=np.uint8), input_size=32)
    assert tuple(image.shape) == (3, 32, 32)
    assert tuple(video.shape) == (2, 3, 32, 32)
