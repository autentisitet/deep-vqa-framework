# src/data/dataset_loaders.py
from dataclasses import dataclass
from typing import Type

from src.utils.registry import Registry

from .metadata_loaders import KonvidLoader, T2VqaLoader, Tid2013Loader


@dataclass(frozen=True)
class DatasetRegistryEntry:
    loader_cls: Type
    metadata_file: str


class MetadataLoaderFactory:
    _REGISTRY = Registry[DatasetRegistryEntry]("metadata_loaders")
    _REGISTRY.register("konvid-1k", DatasetRegistryEntry(KonvidLoader, "KoNViD_1k_mos.csv"))
    _REGISTRY.register("t2vqa-db", DatasetRegistryEntry(T2VqaLoader, "info.txt"))
    _REGISTRY.register("tid2013", DatasetRegistryEntry(Tid2013Loader, "mos_with_names.txt"))

    @classmethod
    def normalize_key(cls, dataset_name: str) -> str:
        key = dataset_name.lower().strip()

        if not cls._REGISTRY.contains(key):
            available = cls._REGISTRY.keys()
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. Available: {available}"
            )

        return key

    @classmethod
    def get_entry(cls, dataset_name: str) -> DatasetRegistryEntry:
        return cls._REGISTRY.get(cls.normalize_key(dataset_name))

    @classmethod
    def get_loader(cls, dataset_name: str):
        return cls.get_entry(dataset_name).loader_cls()

    @classmethod
    def get_metadata_file(cls, dataset_name: str) -> str:
        return cls.get_entry(dataset_name).metadata_file


    @classmethod
    def register(cls, name: str, loader_cls, metadata_file: str = "mos.txt"):
        """动态注册新的数据集加载器（扩展用）"""
        cls._REGISTRY.register(name, DatasetRegistryEntry(loader_cls, metadata_file), replace=True)

    @classmethod
    def available_datasets(cls) -> list:
        """列出所有已注册的数据集名称"""
        return cls._REGISTRY.keys()
