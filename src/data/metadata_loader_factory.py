# src/data/factory.py
from .metadata_loaders import Tid2013Loader, KonvidLoader, T2VqaLoader

class MetadataLoaderFactory:
    # 注册你的 Loader，新增数据集只需在这里加一行
    _REGISTRY = {
        "konvid-1k": KonvidLoader,
        "t2vqa-db": T2VqaLoader,
        "tid2013": Tid2013Loader
    }

    @classmethod
    def get_loader(cls, dataset_name: str):
        key = dataset_name.lower().strip()  # 调用方传入什么都能转成小写
        loader_class = cls._REGISTRY.get(key)
        if not loader_class:
            available = list(cls._REGISTRY.keys())
            raise ValueError(f"❌ 未注册的数据集: {dataset_name}，可用: {available}")

        try:
            return loader_class()
        except Exception as e:
            raise RuntimeError(f"❌ 实例化加载器失败: {dataset_name}, {e}")