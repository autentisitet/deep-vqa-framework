import yaml
from pathlib import Path
from .schemas import Config
from src.data.dataset_loaders import MetadataLoaderFactory


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k not in result:
            result[k] = v
        elif isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    config_dir: Path = Path("config"),
    model_name: str = "resnet_iqa",
    dataset_name: str = "tid2013",
) -> Config:
    dataset_key = MetadataLoaderFactory.normalize_key(dataset_name)

    # 1. 加载所有 YAML
    with open(config_dir / "basic.yaml") as f:
        base_cfg = yaml.safe_load(f) or {}

    with open(config_dir / "dataset_config.yaml") as f:
        all_datasets = yaml.safe_load(f) or {}
        config_dataset_key = next(
            (k for k in all_datasets.keys() if k.lower().strip() == dataset_key),
            None,
        )
        dataset_cfg = all_datasets.get(config_dataset_key, {}) if config_dataset_key else {}
        dataset_cfg["registry_key"] = dataset_key

    with open(config_dir / "models" / f"{model_name}.yaml") as f:
        model_cfg = yaml.safe_load(f) or {}

    # 2. 合并 base + model（model 覆盖 base）
    merged = _deep_merge(base_cfg, model_cfg)

    # 3. dataset 塞进 merged；paths / 其他默认值由 Pydantic 提供
    merged["dataset"] = dataset_cfg

    # 4. 一步到位，Pydantic 自动处理所有嵌套
    return Config(**merged)
