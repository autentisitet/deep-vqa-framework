"""Instance configuration for a Deep-VQA deployment.

Loads ``.env``, ``deploy-config/basic.yaml`` and the selected profile, then
exposes one ``cfg`` object with resolved paths, runtime directories, and the
defaults shared by ``deploy.web`` and ``deploy.inference``.
"""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from deploy.types import AuthMode, EvaluationStoreBackend
from src.config.yaml import deep_merge, load_yaml_mapping
from src.data.dataset_types import DatasetType, TaskType


def _load_project_env() -> None:
    """Load a local .env without overriding explicitly exported variables."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_project_env()


def load_deployment_profile(policy_path: Path) -> dict:
    """Merge deployment basic.yaml with one profiles/ mode override."""
    basic_path = policy_path.parent.parent / "basic.yaml"
    try:
        base = load_yaml_mapping(basic_path)
        policy = load_yaml_mapping(policy_path)
        # Authentication is an explicit mode decision, never a schema default.
        # Otherwise an accidentally empty override could inherit auth.mode=none.
        if not isinstance(policy.get("auth"), dict) or "mode" not in policy["auth"]:
            raise ValueError(f"Deployment profile must explicitly declare auth.mode: {policy_path}")
        return deep_merge(base, policy)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot load deployment configuration: {exc}") from exc


def _load_deployment_settings() -> dict:
    """Load and validate the selected deployment profile YAML."""
    configured = os.getenv("DEEP_VQA_DEPLOYMENT_CONFIG", "").strip()
    if not configured:
        raise RuntimeError(
            "DEEP_VQA_DEPLOYMENT_CONFIG must select deploy-config/profiles/infer_deploy.internal.yaml "
            "or deploy-config/profiles/infer_deploy.proxy.yaml "
            "or deploy-config/profiles/infer_deploy.local.yaml"
        )
    policy_path = Path(configured)
    if not policy_path.is_absolute():
        policy_path = Path(__file__).resolve().parents[1] / policy_path
    if not policy_path.is_file():
        raise RuntimeError(f"Deployment configuration is missing: {policy_path}")
    return load_deployment_profile(policy_path)


class ServiceEndpoints(BaseModel):
    """Central registry for runtime endpoints and local stores."""

    model_config = ConfigDict(extra="forbid")

    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    # This is the host API port used by local tools and tests. The
    # container's internal listener remains fixed by Compose at 8000.
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")), ge=1, le=65535)
    web_host: str = Field(default_factory=lambda: os.getenv("WEB_HOST", "127.0.0.1"))
    web_port: int = Field(default_factory=lambda: int(os.getenv("WEB_PORT", "8000")), ge=1, le=65535)
    sqlite_path: Path = Field(
        default_factory=lambda: Path(os.getenv("DEEP_VQA_SQLITE_PATH", "reports/frontend-evaluations.sqlite3"))
    )
    accounts_sqlite_path: Path = Field(
        default_factory=lambda: Path(os.getenv("DEEP_VQA_ACCOUNTS_SQLITE_PATH", "reports/accounts.sqlite3"))
    )
    upload_dir: Path = Field(default_factory=lambda: Path(os.getenv("DEEP_VQA_UPLOAD_DIR", "uploads")))


class ServingModelSettings(BaseModel):
    """Expected metadata for one configured checkpoint."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: Path
    task_type: TaskType
    media_type: DatasetType
    model_name: str
    backbone: str
    dataset: str
    mos_min: float
    mos_max: float


class InferenceRuntimeSettings(BaseModel):
    """Settings for startup model loading and request serialization."""

    model_config = ConfigDict(extra="forbid")

    preload_models: bool = True
    serialize_heavy_requests: bool = True


class UploadSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_bytes: int = Field(gt=0)
    read_timeout_seconds: float = Field(gt=0)


class SQLiteSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_database_bytes: int = Field(gt=0)
    busy_timeout_seconds: float = Field(gt=0)


class EvaluationStoreSettings(BaseModel):
    """Persistence backend for browser evaluation history."""

    model_config = ConfigDict(extra="forbid")

    backend: EvaluationStoreBackend = "sqlite"


class ArtifactSettings(BaseModel):
    """Retention window for generated visualization artifacts."""

    model_config = ConfigDict(extra="forbid")

    # Expired artifacts are removed opportunistically when a new one is created,
    # never during startup.
    retention_seconds: int = Field(default=604800, gt=0)


class AuthSettings(BaseModel):
    """Deployment-level access control and login abuse limits."""

    model_config = ConfigDict(extra="forbid")

    mode: AuthMode = AuthMode.NONE
    login_attempts: int = Field(default=5, ge=1, le=100)
    login_window_seconds: int = Field(default=900, ge=60, le=86400)
    session_max_age_seconds: int = Field(default=86400, ge=300, le=604800)


class DeploymentSettings(BaseModel):
    """Repository-managed model, upload, storage, and login settings."""

    model_config = ConfigDict(extra="forbid")

    models: dict[str, ServingModelSettings]
    inference_runtime: InferenceRuntimeSettings = Field(default_factory=InferenceRuntimeSettings)
    uploads: UploadSettings
    sqlite: SQLiteSettings
    evaluation_store: EvaluationStoreSettings = Field(default_factory=EvaluationStoreSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)


class InferenceConfig(BaseModel):
    """Deployment runtime defaults only."""

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    endpoints: ServiceEndpoints = Field(default_factory=ServiceEndpoints)
    deployment: DeploymentSettings = Field(
        default_factory=lambda: DeploymentSettings.model_validate(_load_deployment_settings())
    )
    deploy_dir: Path = Path("deploy")
    reports_dir: Path = Path("reports")
    examples_dir: Path = Path("examples")

    iqa_model_path: Path = Field(default=Path("deploy/iqa-models/tid2013_best.pt"))
    vqa_model_path: Path = Field(default=Path("deploy/vqa-models/konvid-1k_best.pt"))

    num_frames: int = Field(default=8)
    input_size: int = Field(default=224)

    image_exts: set = Field(
        default_factory=lambda: DatasetType.extensions_for("image") & DatasetType.stable_extensions()
    )
    video_exts: set = Field(
        default_factory=lambda: DatasetType.extensions_for("video") & DatasetType.stable_extensions()
    )

    black_threshold: float = Field(default=8.0)
    white_threshold: float = Field(default=245.0)
    max_black_white_ratio: float = Field(default=0.3)
    max_frame_deviation: float = Field(default=0.10)

    default_device: str = Field(default="cuda" if __import__("torch").cuda.is_available() else "cpu")

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to the project root when it is not absolute."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.project_root / candidate


cfg = InferenceConfig()


def _ensure_writable_directory(path: Path) -> None:
    """Create a runtime directory and fail startup if it cannot accept files."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    probe_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=".deep-vqa-write-", dir=path)
        os.close(descriptor)
        probe_path = Path(raw_path)
    except OSError as exc:
        raise RuntimeError(f"Runtime directory is not writable: {path}") from exc
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def ensure_runtime_dirs(config: InferenceConfig = cfg) -> None:
    """Create runtime directories and verify the upload directory is writable."""
    for path in (
        config.resolve_path(config.examples_dir / "images"),
        config.resolve_path(config.examples_dir / "videos"),
        config.resolve_path(config.deploy_dir / "iqa-models"),
        config.resolve_path(config.deploy_dir / "vqa-models"),
        config.resolve_path(config.reports_dir / "iqa-test"),
        config.resolve_path(config.reports_dir / "vqa-test"),
    ):
        path.mkdir(parents=True, exist_ok=True)
    _ensure_writable_directory(config.resolve_path(config.endpoints.upload_dir))
