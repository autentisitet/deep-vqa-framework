# src/utils/config_loader.py
import yaml
from pathlib import Path
from loguru import logger
from collections.abc import Mapping
import pdb

# 改为自动扫描，或者支持动态注册
def discover_models(config_dir: Path) -> dict:
    """自动发现 models 目录下的配置文件"""
    models_dir = config_dir / "models"
    if not models_dir.exists():
        return {}

    model_map = {}
    for yaml_file in models_dir.glob("*.yaml"):
        # resnet_iqa.yaml -> resnet_iqa
        model_name = yaml_file.stem
        model_map[model_name] = f"models/{yaml_file.name}"
    return model_map

MODEL_MAP = None  # 延迟加载



def get_model_map(config_dir: Path) -> dict:
    global MODEL_MAP
    if MODEL_MAP is None:
        MODEL_MAP = discover_models(config_dir)
        # 保留手动映射作为后备
        MODEL_MAP.update({
            'resnet_iqa': 'models/resnet_iqa.yaml',
            'timeswin_vqa': 'models/timeswin_vqa.yaml'
        })
    return MODEL_MAP



def deep_update(source, overrides):
    """递归深度合并字典"""
    for k, v in overrides.items():
        if isinstance(v, Mapping) and v:
            source[k] = deep_update(source.get(k, {}), v)
        else:
            source[k] = v
    return source



def safe_load_yaml(path: Path, description: str = "配置文件") -> dict:
    """安全加载 YAML 文件，失败时抛出异常"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError(f"{description} 文件为空: {path}")
            return content
    except FileNotFoundError as e:
        logger.error(f"❌ {description} 不存在: {path}")
        raise FileNotFoundError(f"{description} 不存在: {path}") from e
    except yaml.YAMLError as e:
        logger.error(f"❌ {description} YAML 格式错误: {path}")
        logger.error(f"   错误详情: {e}")
        # Help: 检查 YAML 语法，特别是缩进和特殊字符
        raise RuntimeError(f"{description} YAML 格式错误: {path}") from e
    except Exception as e:
        logger.error(f"❌ 读取 {description} 失败: {path}, 错误: {e}")
        raise




def load_system_config(model_cfg_name: str, dataset_name: str) -> dict:
    config_dir = Path("config")

    # 1. 加载基础配置
    basic_path = config_dir / "basic.yaml"
    config = safe_load_yaml(basic_path, "基础配置文件")

    # 2. 加载模型配置
    model_key = str(model_cfg_name).strip().lower()
    model_map = get_model_map(config_dir)
    target_model_file = model_map.get(model_key, 'models/resnet_iqa.yaml')
    model_path = config_dir / target_model_file

    if not model_path.exists():
        logger.warning(f"⚠️ Model config [{model_path}] not found. Falling back to resnet_iqa.yaml")
        model_path = config_dir / 'models/resnet_iqa.yaml'
        # 如果 fallback 也不存在，会在 safe_load_yaml 中报错

    model_config = safe_load_yaml(model_path, f"模型配置文件 [{model_key}]")
    config = deep_update(config, model_config)

    # 3. 加载数据集配置
    dataset_cfg_path = config_dir / "dataset_config.yaml"
    ds_all = safe_load_yaml(dataset_cfg_path, "数据集配置文件")

    ds_all_lowered = {k.lower(): v for k, v in ds_all.items()}
    target_ds_key = str(dataset_name).strip().lower()

    if target_ds_key not in ds_all_lowered:
        available = list(ds_all.keys())
        logger.error(f"❌ Dataset '{dataset_name}' not found in dataset_config.yaml")
        logger.info(f"   可用数据集: {available}")
        debug_breakpoint()
        raise KeyError(f"Dataset settings for '{dataset_name}' missing. Available: {available}")

    dataset_info = ds_all_lowered[target_ds_key]

    # ✅ 修复：命令行参数优先，不使用 YAML 中的 name
    config['dataset_info'] = dataset_info
    config['dataset_name'] = dataset_name.lower()  # 命令行参数转小写

    logger.info(f"⚙️ [Config Engine] Layered configuration successfully built for "
                f"Model [{model_key}] & Dataset [{config['dataset_name']}]")

    logger.debug(f"[Config] 模型配置: {model_key}, 数据集: {config['dataset_name']}")
    logger.debug(f"[Config] 训练配置: epochs={config.get('train', {}).get('epochs', 'N/A')}")

    return config