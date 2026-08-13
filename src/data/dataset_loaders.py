# src/data/dataset_loaders.py
from dataclasses import dataclass
from typing import Type

from .metadata_loaders import KonvidLoader, T2VqaLoader, Tid2013Loader


@dataclass(frozen=True)
class DatasetRegistryEntry:
    loader_cls: Type
    metadata_file: str


class MetadataLoaderFactory:
    # Register the Loader; to add a new dataset, simply add one line here.
    _REGISTRY = {
        "konvid-1k": DatasetRegistryEntry(KonvidLoader, "KoNViD_1k_mos.csv"),
        "t2vqa-db": DatasetRegistryEntry(T2VqaLoader, "info.txt"),
        "tid2013": DatasetRegistryEntry(Tid2013Loader, "mos_with_names.txt"),
    }

    @classmethod
    def normalize_key(cls, dataset_name: str) -> str:
        key = dataset_name.lower().strip()

        if key not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. Available: {available}"
            )

        return key

    @classmethod
    def get_entry(cls, dataset_name: str) -> DatasetRegistryEntry:
        return cls._REGISTRY[cls.normalize_key(dataset_name)]

    @classmethod
    def get_loader(cls, dataset_name: str):
        return cls.get_entry(dataset_name).loader_cls()

    @classmethod
    def get_metadata_file(cls, dataset_name: str) -> str:
        return cls.get_entry(dataset_name).metadata_file


    @classmethod
    def register(cls, name: str, loader_cls, metadata_file: str = "mos.txt"):
        """动态注册新的数据集加载器（扩展用）"""
        cls._REGISTRY[name.lower().strip()] = DatasetRegistryEntry(loader_cls, metadata_file)

    @classmethod
    def available_datasets(cls) -> list:
        """列出所有已注册的数据集名称"""
        return list(cls._REGISTRY.keys())
