# src/data/__init__.py
from importlib import import_module

__all__ = [
    "DataEDA",
    "MetadataLoaderFactory",
    "DatasetType",
    "BaseMetadataLoader",
    "Tid2013Loader",
    "KonvidLoader",
    "T2VqaLoader",
    "PreprocessingAction",
    "PREPROCESSING_REGISTRY",
    "preprocessing_actions",
]

_LAZY_IMPORTS = {
    "DataEDA": ".data_eda",
    "MetadataLoaderFactory": ".dataset_loaders",
    "DatasetType": ".dataset_types",
    "BaseMetadataLoader": ".metadata_loaders",
    "Tid2013Loader": ".metadata_loaders",
    "KonvidLoader": ".metadata_loaders",
    "T2VqaLoader": ".metadata_loaders",
    "PreprocessingAction": ".preprocessing",
    "PREPROCESSING_REGISTRY": ".preprocessing",
    "preprocessing_actions": ".preprocessing",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
