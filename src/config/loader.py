import yaml
from pathlib import Path
from .schemas import Config


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
    # 1. 加载所有 YAML
    with open(config_dir / "basic.yaml") as f:
        base_cfg = yaml.safe_load(f) or {}

    with open(config_dir / "dataset_config.yaml") as f:
        all_datasets = yaml.safe_load(f) or {}
        dataset_key = next(
            (k for k in all_datasets.keys() if k.lower() == dataset_name.lower()),
            None
        )
        dataset_cfg = all_datasets.get(dataset_key, {}) if dataset_key else {}

    with open(config_dir / "models" / f"{model_name}.yaml") as f:
        model_cfg = yaml.safe_load(f) or {}

    # 2. 合并 base + model（model 覆盖 base）
    merged = _deep_merge(base_cfg, model_cfg)

    # 3. 手动构造 paths（不从 YAML 读）
    merged["paths"] = {
        "project_root": str(Path(".").resolve()),
        "datasets_dir": "datasets",
        "results_dir": "results",
        "logs_dir": "logs",
        "cache_dir": ".download_cache",
    }

    # 4. dataset 塞进 merged（paths 由 Pydantic 默认值提供）
    merged["dataset"] = dataset_cfg

    # 5. 一步到位，Pydantic 自动处理所有嵌套
    return Config(**merged)