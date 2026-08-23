from pathlib import Path

import os

from pydantic import BaseModel, ConfigDict, Field
import yaml
from src.data.dataset_types import DatasetType


def _load_project_env() -> None:
    """Load a local .env without overriding explicitly exported variables."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


_load_project_env()


def _load_deployment_policy() -> dict:
    configured = os.getenv("DEEP_VQA_DEPLOYMENT_CONFIG", "").strip()
    if not configured:
        raise RuntimeError(
            "DEEP_VQA_DEPLOYMENT_CONFIG must select deploy-config/profiles/infer_deploy.internal.yaml "
            "or deploy-config/profiles/infer_deploy.public.yaml"
        )
    policy_path = Path(configured)
    if not policy_path.is_absolute():
        policy_path = Path(__file__).resolve().parents[2] / policy_path
    if not policy_path.is_file():
        raise RuntimeError(f"Deployment policy is missing: {policy_path}")
    with policy_path.open(encoding="utf-8") as handle:
        policy = yaml.safe_load(handle) or {}
    if not isinstance(policy, dict):
        raise RuntimeError(f"Deployment policy must be a mapping: {policy_path}")
    return policy


class ServiceEndpoints(BaseModel):
    """Central registry for runtime endpoints and local stores."""

    model_config = ConfigDict(extra="forbid")

    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    # This is the host-facing API port used by local tooling and tests. The
    # container's internal listener remains fixed by Compose at 8000.
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")), ge=1, le=65535)
    web_host: str = Field(default_factory=lambda: os.getenv("WEB_HOST", "127.0.0.1"))
    web_port: int = Field(default_factory=lambda: int(os.getenv("WEB_PORT", "8000")), ge=1, le=65535)
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    sqlite_path: Path = Field(default_factory=lambda: Path(os.getenv("DEEP_VQA_SQLITE_PATH", "reports/frontend-evaluations.sqlite3")))


class OllamaPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_model: str
    timeout_seconds: float = Field(gt=0)


class UploadPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_bytes: int = Field(gt=0)
    read_timeout_seconds: float = Field(gt=0)


class SQLitePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_database_bytes: int = Field(gt=0)
    busy_timeout_seconds: float = Field(gt=0)


class EvaluationStorePolicy(BaseModel):
    """Persistence backend for browser evaluation history."""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(default="sqlite", pattern="^(sqlite|none)$")


class AuthPolicy(BaseModel):
    """Deployment-level access control."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(default="none", pattern="^(none|api_key)$")


class DeploymentPolicy(BaseModel):
    """Stable, repository-managed serving limits and model policy."""

    model_config = ConfigDict(extra="forbid")

    ollama: OllamaPolicy
    uploads: UploadPolicy
    sqlite: SQLitePolicy
    evaluation_store: EvaluationStorePolicy = Field(default_factory=EvaluationStorePolicy)
    auth: AuthPolicy = Field(default_factory=AuthPolicy)


class InferenceConfig(BaseModel):
    """Deployment runtime defaults only."""

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    endpoints: ServiceEndpoints = Field(default_factory=ServiceEndpoints)
    deployment: DeploymentPolicy = Field(default_factory=lambda: DeploymentPolicy.model_validate(_load_deployment_policy()))
    deploy_dir: Path = Path("deploy")
    reports_dir: Path = Path("reports")
    examples_dir: Path = Path("examples")

    iqa_model_path: Path = Field(default=Path("deploy/iqa-models/tid2013_best.pt"))
    vqa_model_path: Path = Field(default=Path("deploy/vqa-models/konvid-1k_best.pt"))

    num_frames: int = Field(default=8)
    input_size: int = Field(default=224)

    image_exts: set = Field(default_factory=lambda: DatasetType.extensions_for("image") & DatasetType.stable_extensions())
    video_exts: set = Field(default_factory=lambda: DatasetType.extensions_for("video") & DatasetType.stable_extensions())

    black_threshold: float = Field(default=8.0)
    white_threshold: float = Field(default=245.0)
    max_black_white_ratio: float = Field(default=0.3)
    max_frame_deviation: float = Field(default=0.10)

    default_device: str = Field(default="cuda" if __import__("torch").cuda.is_available() else "cpu")

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.project_root / candidate


cfg = InferenceConfig()


def ensure_runtime_dirs() -> None:
    for path in (
        cfg.resolve(cfg.examples_dir / "images"),
        cfg.resolve(cfg.examples_dir / "videos"),
        cfg.resolve(cfg.deploy_dir / "iqa-models"),
        cfg.resolve(cfg.deploy_dir / "vqa-models"),
        cfg.resolve(cfg.reports_dir / "iqa-test"),
        cfg.resolve(cfg.reports_dir / "vqa-test"),
    ):
        path.mkdir(parents=True, exist_ok=True)
