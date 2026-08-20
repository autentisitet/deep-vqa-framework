# src/config/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from pathlib import Path


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ============================================================
# 路径配置
# ============================================================
class PathsConfig(_BaseModel):
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    datasets_dir: Path = Path("datasets")
    results_dir: Path = Path("results")
    logs_dir: Path = Path("logs")
    deploy_dir: Path = Path("deploy")
    reports_dir: Path = Path("reports")
    examples_dir: Path = Path("examples")
    cache_dir: Path = Path(".cache")

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    def dataset_slug(self, dataset_name: str) -> str:
        """Normalize dataset output folder names under results/."""
        return str(dataset_name).strip().lower()

    def dataset_results_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/"""
        return self.resolve(self.results_dir) / self.dataset_slug(dataset_name)

    def train_logs_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/train_logs/"""
        return self.dataset_results_dir(dataset_name) / "train_logs"

    def plots_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/plots/"""
        return self.dataset_results_dir(dataset_name) / "plots"

    def eda_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/eda/"""
        return self.dataset_results_dir(dataset_name) / "eda"

    def feature_distribution_dir(self, dataset_name: str) -> Path:
        """Directory for saved training feature-distribution artifacts."""
        return self.eda_dir(dataset_name) / "feature_distribution"

    def model_outputs_dir(self, dataset_name: str) -> Path:
        """results/{dataset_name}/model_outputs/"""
        return self.dataset_results_dir(dataset_name) / "model_outputs"

    def corrupted_dir(self, dataset_name: str) -> Path:
        return self.dataset_results_dir(dataset_name) / "corrupted"

    def deploy_iqa_dir(self) -> Path:
        return self.resolve(self.deploy_dir) / "iqa-models"

    def deploy_vqa_dir(self) -> Path:
        return self.resolve(self.deploy_dir) / "vqa-models"

    def reports_iqa_dir(self) -> Path:
        return self.resolve(self.reports_dir) / "iqa-test"

    def reports_vqa_dir(self) -> Path:
        return self.resolve(self.reports_dir) / "vqa-test"

# ============================================================
# 系统配置
# ============================================================
class SystemConfig(_BaseModel):
    amp: bool = True
    pin_memory: bool = True


# ============================================================
# 预处理配置
# ============================================================
class PreprocessingConfig(_BaseModel):
    seed: int = 42
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
    monitor: str = "val_srocc"
    secondary_monitor: str = "val_plcc"
    mode: str = "max"
    save_top_k: int = 2


# ============================================================
# 训练配置
# ============================================================
class ManifestConfig(_BaseModel):
    """Controls which high-error predictions are written to manifests."""

    enabled: bool = True
    thresholds: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5])


class TrainConfig(_BaseModel):
    epochs: int = 50
    lr: float = 0.0001
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    gradient_accumulation_steps: int = 4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    manifest: ManifestConfig = Field(default_factory=ManifestConfig)
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


# ============================================================
# 损失配置
# ============================================================
class LossConfig(_BaseModel):
    smooth_l1_weight: float = 0.7
    rank_weight: float = 0.3
    huber_delta: float = 0.1
    rank_epsilon: float = 0.01
    max_pairs: int = 5000


# ============================================================
# 模型配置
# ============================================================
class ModelArchConfig(_BaseModel):
    name: str
    backbone: str
    image_backbone: Optional[str] = None
    video_backbone: str = "swin_t"
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


# ============================================================
# 数据集配置
# ============================================================
class DatasetPathsConfig(_BaseModel):
    root: str
    data: str
    metadata: str


class DatasetMetadataConfig(_BaseModel):
    mos_file: Optional[str] = None


class DatasetMetaConfig(_BaseModel):
    name: str
    task_type: str  # "iqa" or "vqa"
    data_type: str  # "image" or "video"
    paths: DatasetPathsConfig
    registry_key: str = ""
    metadata: DatasetMetadataConfig = Field(default_factory=DatasetMetadataConfig)
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
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetMetaConfig

    # 模型相关
    task_type: str                    # "iqa" or "vqa"
    model: ModelArchConfig            # 模型架构
    loss: LossConfig                  # 损失配置
