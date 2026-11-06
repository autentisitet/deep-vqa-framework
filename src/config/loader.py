from pathlib import Path

from .schemas import Config
from .yaml import deep_merge, load_yaml_mapping


def load_config(
    config_dir: Path = Path("train-config"),
    model_name: str = "swin_iqa",
    dataset_name: str = "tid2013",
) -> Config:
    """Assemble and validate a ``Config`` from the training YAML files."""
    # Import lazily: dataset_loaders imports src.utils, whose package exports
    # config symbols. Keeping this import at module scope creates a cycle
    # during test/module discovery.
    from src.data.dataset_loaders import MetadataLoaderFactory

    dataset_key = MetadataLoaderFactory.normalize_key(dataset_name)

    base_cfg = load_yaml_mapping(config_dir / "basic.yaml")
    all_datasets = load_yaml_mapping(config_dir / "dataset_config.yaml")
    config_dataset_key = next((key for key in all_datasets if key.lower().strip() == dataset_key), None)
    dataset_cfg = dict(all_datasets.get(config_dataset_key, {})) if config_dataset_key else {}
    dataset_cfg["registry_key"] = dataset_key
    model_cfg = load_yaml_mapping(config_dir / "models" / f"{model_name}.yaml")

    # 2. 合并 base + model（model 覆盖 base）
    merged = deep_merge(base_cfg, model_cfg)

    # 3. dataset 塞进 merged；paths / 其他默认值由 Pydantic 提供
    merged["dataset"] = dataset_cfg

    # 4. 一步到位，Pydantic 自动处理所有嵌套
    return Config(**merged)
