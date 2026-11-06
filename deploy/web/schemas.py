"""HTTP request and response models for the deployment API.

Kept separate from ``deploy/web/api.py`` so route handlers stay readable and the
wire contract has a single place to live.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from deploy.types import AuthMode
from deploy.web.exceptions import ErrorCode
from src.data.dataset_types import DatasetType, TaskType


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"code": 1100, "error": "Model 'iqa' is not configured", "detail": None}}
    )

    code: ErrorCode = Field(description="Stable business error code outside the reserved HTTP status range 100-599")
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    device: str


class ModelSummary(BaseModel):
    model_id: str = Field(description="Configured model identifier: iqa or vqa")
    loaded: bool = Field(description="Whether this model currently occupies worker memory")
    task_type: TaskType
    model_name: str
    backbone: str
    dataset: str
    media_type: DatasetType
    mos_min: float
    mos_max: float
    checkpoint: str


class ModelCollectionResponse(BaseModel):
    items: list[ModelSummary]
    count: int


class EvaluationResponse(BaseModel):
    evaluation_id: str
    filename: str
    model_id: str
    media_type: DatasetType
    task_type: TaskType
    model_name: str
    backbone: str
    dataset: str
    score: float = Field(description="Normalized predicted quality score")
    mos_score: float | None = Field(description="Score converted to dataset MOS range")
    mos_min: float
    mos_max: float
    media_size: str
    latency_ms: float


class VisualizationResponse(BaseModel):
    visualization_id: str
    model_id: str
    filename: str
    score: float
    layers: list[str]
    feature_maps: dict[str, str]
    gradcam: str | None = None


class FrontendEvaluationRecord(BaseModel):
    """Public history fields; owner/session identifiers stay in storage."""

    timestamp: str
    file_name: str
    file_hash: str
    task_type: TaskType
    model_used: str
    mos_score: float | None
    mos_interval: tuple[float, float]
    inference_time_ms: float = Field(ge=0)


class FrontendEvaluationLogResponse(BaseModel):
    """Frontend evaluation records read from the SQLite evaluation store."""

    items: list[FrontendEvaluationRecord]
    count: int


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # api_key is used by the shared-key mode; email/password by the accounts mode.
    # Exactly one shape is accepted, depending on the selected auth mode.
    api_key: str | None = None
    email: str | None = None
    password: str | None = None


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class AuthStatusResponse(BaseModel):
    required: bool
    authenticated: bool
    mode: AuthMode = AuthMode.API_KEY
    email: str | None = None


class FrontendSettingsResponse(BaseModel):
    """Frontend login, history, and visualization settings."""

    auth_required: bool
    authenticated: bool
    auth_mode: AuthMode = AuthMode.API_KEY
    email: str | None = None
    evaluation_history_enabled: bool
    visualizations_enabled: bool


__all__ = [
    "AuthStatusResponse",
    "ErrorResponse",
    "EvaluationResponse",
    "FrontendEvaluationLogResponse",
    "FrontendEvaluationRecord",
    "FrontendSettingsResponse",
    "HealthResponse",
    "LoginRequest",
    "ModelCollectionResponse",
    "ModelSummary",
    "RegisterRequest",
    "VisualizationResponse",
]
