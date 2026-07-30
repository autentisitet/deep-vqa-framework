# src/config/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from pathlib import Path


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ============================================================
# 路径配置
# ============================================================
class PathsConfig(_BaseModel):
    project_root: Path = Path(".")
    datasets_dir: Path = Path("datasets")
    results_dir: Path = Path("results")
    logs_dir: Path = Path("logs")
    cache_dir: Path = Path(".download_cache")

    def dataset_results_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/"""
        return self.results_dir / dataset_name

    def train_logs_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/train_logs/"""
        return self.dataset_results_dir(dataset_name) / "train_logs"

    def plots_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/plots/"""
        return self.dataset_results_dir(dataset_name) / "plots"

    def model_outputs_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/model_outputs/"""
        return self.dataset_results_dir(dataset_name) / "model_outputs"

    def corrupted_dir(self, dataset_name: str) -> Path:
        return self.results_dir / dataset_name / "corrupted"

# ============================================================
# 系统配置
# ============================================================
class SystemConfig(_BaseModel):
    project_name: str = "deep-vqa-framework"
    env: str = "autodl"
    device: str = "cuda"
    amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True


# ============================================================
# 预处理配置
# ============================================================
class PreprocessingConfig(_BaseModel):
    seed: int = 42
    shuffle: bool = True
    pin_memory: bool = True
    val_interval: int = 1
    k_fold: int = 5
    batch_size: int = 32
    num_workers: int = 4


# ============================================================
# 早停和检查点配置
# ============================================================
class EarlyStopConfig(_BaseModel):
    enabled: bool = True
    patience: int = 10
    monitor: str = "val_srocc"
    mode: str = "max"


class CheckpointConfig(_BaseModel):
    save_best: bool = True
    save_last: bool = True
    monitor: str = "val_srocc"
    mode: str = "max"
    save_top_k: int = 3


# ============================================================
# 训练配置
# ============================================================
class TrainConfig(_BaseModel):
    epochs: int = 50
    lr: float = 0.0001
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    gradient_accumulation_steps: int = 4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


# ============================================================
# 损失配置
# ============================================================
class LossConfig(_BaseModel):
    mse_weight: float = 0.7
    rank_weight: float = 0.3
    plcc_weight: float = 0.0
    max_pairs: int = 5000


# ============================================================
# 模型配置
# ============================================================
class ModelArchConfig(_BaseModel):
    name: str
    backbone: str
    freeze_backbone: bool = False
    pretrained: bool = True
    dropout: float = 0.3
    input_size: int = 224
    num_frames: int = 8
    transformer_layers: Optional[int] = None  # VQA 用


# ============================================================
# 日志配置
# ============================================================
class LoggingConfig(_BaseModel):
    log_interval: int = 10
    save_period: int = 1
    tensorboard: bool = True
    log_level: str = "INFO"
    print_frequency: int = 50
    save_sample_images: bool = False


# ============================================================
# 评估配置
# ============================================================
class EvaluationConfig(_BaseModel):
    test_after_training: bool = True
    test_best_epoch: bool = True
    save_predictions: bool = True
    metrics: List[str] = ["plcc", "srocc", "krocc", "rmse"]


# ============================================================
# 数据集配置
# ============================================================
class DatasetPathsConfig(_BaseModel):
    root: str
    data: str
    metadata: str


class DatasetMetaConfig(_BaseModel):
    name: str
    task_type: str  # "iqa" or "vqa"
    data_type: str  # "image" or "video"
    paths: DatasetPathsConfig
    mos_min: float = 0.0
    mos_max: float = 5.0


# ============================================================
# 总配置
# ============================================================
class Config(_BaseModel):
    """总配置入口"""
    system: SystemConfig
    preprocessing: PreprocessingConfig
    logging: LoggingConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    dataset: DatasetMetaConfig

    # 模型相关
    task_type: str                    # "iqa" or "vqa"
    model: ModelArchConfig            # 模型架构
    loss: LossConfig                  # 损失配置