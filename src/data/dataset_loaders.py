# src/data/dataset_loaders.py
from .metadata_loaders import KonvidLoader, T2VqaLoader, Tid2013Loader


class MetadataLoaderFactory:
    # Register the Loader; to add a new dataset, simply add one line here.
    _REGISTRY = {
        "konvid-1k": KonvidLoader,
        "t2vqa-db": T2VqaLoader,
        "tid2013": Tid2013Loader,
    }

    @classmethod
    def get_loader(cls, dataset_name: str):
        key = dataset_name.lower().strip()  # Convert to lowercase
        loader_cls = cls._REGISTRY.get(key)

        if loader_cls is None:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. Available: {available}"
            )

        return loader_cls()


    @classmethod
    def register(cls, name: str, loader_cls):
        """动态注册新的数据集加载器（扩展用）"""
        cls._REGISTRY[name.lower()] = loader_cls

    @classmethod
    def available_datasets(cls) -> list:
        """列出所有已注册的数据集名称"""
        return list(cls._REGISTRY.keys())