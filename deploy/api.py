"""REST API for image and video quality assessment.

The service exposes OpenAPI documentation automatically:

* Swagger UI: ``/docs``
* ReDoc: ``/redoc``
* OpenAPI document: ``/openapi.json``
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from src import __version__

from deploy.core.inference import predict_single
from deploy.core.model_loader import load_checkpoint
from deploy.core.runtime_config import cfg, ensure_runtime_dirs
from deploy.core.subjective_quality import SubjectiveQualityError, assess_subjective_quality
from deploy.core.evaluation_store import EvaluationRecord, SQLiteEvaluationStore
from src.visualization.feature_visualizer import FeatureVisualizer, load_image_tensor


API_PREFIX = "/v1"
DEVICE = cfg.default_device
SAFE_ARTIFACT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
MODEL_PATHS = {
    "iqa": cfg.resolve(cfg.iqa_model_path),
    "vqa": cfg.resolve(cfg.vqa_model_path),
}
OPENAPI_OUTPUT_PATH = cfg.resolve(Path("docs/openapi.json"))
FRONTEND_EVALUATION_DB_PATH = cfg.resolve(cfg.endpoints.sqlite_path)
LEGACY_FRONTEND_EVALUATION_LOG_PATH = cfg.resolve(cfg.reports_dir / "frontend-evaluations.jsonl")

OLLAMA_BASE_URL = cfg.endpoints.ollama_base_url
OLLAMA_QUALITY_MODEL = cfg.deployment.ollama.quality_model
OLLAMA_TIMEOUT_SECONDS = cfg.deployment.ollama.timeout_seconds
MAX_UPLOAD_BYTES = cfg.deployment.uploads.max_bytes
UPLOAD_READ_TIMEOUT_SECONDS = cfg.deployment.uploads.read_timeout_seconds
SQLITE_BUSY_TIMEOUT_SECONDS = cfg.deployment.sqlite.busy_timeout_seconds
SQLITE_MAX_DATABASE_BYTES = cfg.deployment.sqlite.max_database_bytes
EVALUATION_STORE_BACKEND = cfg.deployment.evaluation_store.backend
AUTH_MODE = cfg.deployment.auth.mode
API_KEY = os.getenv("DEEP_VQA_API_KEY", "")
AUTH_COOKIE = "deep_vqa_auth"
AUTH_TOKEN = hashlib.pbkdf2_hmac(
    "sha256",
    API_KEY.encode(),
    (os.getenv("DEEP_VQA_AUTH_SECRET", "").encode() or secrets.token_bytes(32)),
    310000,
).hex()
FRONTEND_SESSION_HEADER = "X-Deep-VQA-Session"
LEGACY_SESSION_ID = "legacy"
_frontend_log_lock = threading.Lock()
_evaluation_store = SQLiteEvaluationStore(
    FRONTEND_EVALUATION_DB_PATH, SQLITE_MAX_DATABASE_BYTES, SQLITE_BUSY_TIMEOUT_SECONDS
)


class UploadLimitError(ValueError):
    """Raised when an uploaded media file exceeds the configured limits."""


class UploadTimeoutError(TimeoutError):
    """Raised when reading an uploaded media file takes too long."""


class ErrorResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"error": "Model 'iqa' is not loaded"}})

    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    device: str


class ModelSummary(BaseModel):
    model_id: str = Field(description="Serving role: iqa or vqa")
    task_type: str
    model_name: str
    backbone: str
    dataset: str
    media_type: str
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
    media_type: str
    task_type: str
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


class SubjectiveAssessmentResponse(BaseModel):
    """Ephemeral Ollama-based quality description and Bayesian assessment."""

    media_type: str
    description: str
    llm_score: int = Field(description="Ollama subjective visual-quality score in [0, 100]")
    bayesian_score: float = Field(description="Beta-prior posterior mean in [0, 100]")
    credible_interval: tuple[float, float] = Field(description="Approximate 95% Bayesian credible interval")
    model_used: str
    frames_sampled: int


class FrontendEvaluationLogResponse(BaseModel):
    """Frontend evaluation records read from the SQLite evaluation store."""

    items: list[dict[str, Any]]
    count: int


class LoginRequest(BaseModel):
    api_key: str


class AuthStatusResponse(BaseModel):
    required: bool
    authenticated: bool


def _auth_required() -> bool:
    return AUTH_MODE == "api_key"


def _is_authenticated(request: Request) -> bool:
    if not _auth_required():
        return True
    if not API_KEY:
        return False
    supplied = request.cookies.get(AUTH_COOKIE, "")
    return hmac.compare_digest(supplied, AUTH_TOKEN)


def _require_auth(request: Request) -> None:
    if not _is_authenticated(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


model_cache: dict[str, dict[str, Any]] = {}


def _error(message: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error=message, detail=detail).model_dump(),
    )


def _mos_range(config: Any) -> tuple[float, float]:
    mos_min = float(config.dataset.mos_min)
    mos_max = float(config.dataset.mos_max)
    if mos_max <= mos_min:
        raise ValueError(f"Invalid checkpoint MOS range: [{mos_min}, {mos_max}]")
    return mos_min, mos_max


def _dataset_name(config: Any) -> str:
    dataset = config.dataset
    return str(getattr(dataset, "name", None) or getattr(dataset, "registry_key", "unknown"))


def _frontend_session_id(request: Request) -> str:
    """Return the browser session identifier used to isolate history records."""
    value = request.headers.get(FRONTEND_SESSION_HEADER, "").strip()
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"A valid {FRONTEND_SESSION_HEADER} header is required",
        ) from exc


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of an uploaded media file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frontend_evaluation_connection() -> sqlite3.Connection:
    _evaluation_store.path = FRONTEND_EVALUATION_DB_PATH
    _evaluation_store.max_bytes = SQLITE_MAX_DATABASE_BYTES
    _evaluation_store.busy_timeout_seconds = SQLITE_BUSY_TIMEOUT_SECONDS
    return _evaluation_store.connection()


async def _save_upload_to_temp(file: UploadFile, suffix: str) -> str:
    """Copy an upload with explicit byte and read-time limits."""
    if MAX_UPLOAD_BYTES <= 0:
        raise ValueError("uploads.max_bytes in the selected deployment profile must be positive")
    if UPLOAD_READ_TIMEOUT_SECONDS <= 0:
        raise ValueError("uploads.read_timeout_seconds in the selected deployment profile must be positive")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    total_bytes = 0
    try:
        with os.fdopen(tmp_fd, "wb") as target:
            while True:
                try:
                    chunk = await asyncio.wait_for(file.read(1024 * 1024), timeout=UPLOAD_READ_TIMEOUT_SECONDS)
                except asyncio.TimeoutError as exc:
                    raise UploadTimeoutError(
                        f"Upload read exceeded {UPLOAD_READ_TIMEOUT_SECONDS:g} seconds"
                    ) from exc
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise UploadLimitError(
                        f"Uploaded file exceeds the {MAX_UPLOAD_BYTES} byte limit"
                    )
                target.write(chunk)
        return tmp_path
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _initialize_frontend_evaluation_store() -> None:
    """Create the SQLite store and import a legacy JSONL log once when needed."""
    if EVALUATION_STORE_BACKEND == "none":
        return
    _evaluation_store.path = FRONTEND_EVALUATION_DB_PATH
    _evaluation_store.initialize(LEGACY_FRONTEND_EVALUATION_LOG_PATH)
    return

def _append_frontend_evaluation_log(
    response: EvaluationResponse, media_path: Path, session_id: str = LEGACY_SESSION_ID
) -> None:
    """Store one successful browser evaluation in SQLite."""
    if EVALUATION_STORE_BACKEND == "none":
        return
    _evaluation_store.path = FRONTEND_EVALUATION_DB_PATH
    record = EvaluationRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_name=response.filename,
        file_hash=_sha256(media_path),
        task_type=response.task_type,
        model_used=response.model_name,
        mos_score=response.mos_score,
        mos_min=response.mos_min,
        mos_max=response.mos_max,
        inference_time_ms=response.latency_ms,
    )
    # Tests and lightweight callers may append before FastAPI lifespan runs;
    # ensure the schema exists at the persistence boundary as well.
    _evaluation_store.initialize()
    _evaluation_store.append(record)


def _read_frontend_evaluation_log(limit: int, session_id: str | None = None) -> list[dict[str, Any]]:
    """Return the newest frontend evaluation records from SQLite."""
    if EVALUATION_STORE_BACKEND == "none":
        return []
    _evaluation_store.path = FRONTEND_EVALUATION_DB_PATH
    _evaluation_store.initialize(LEGACY_FRONTEND_EVALUATION_LOG_PATH)
    return _evaluation_store.read(limit, session_id)


def _load_model(model_id: str, checkpoint_path: Path) -> None:
    model, config = load_checkpoint(checkpoint_path, device=DEVICE)
    mos_min, mos_max = _mos_range(config)
    model_cache[model_id] = {
        "model": model,
        "config": config,
        "mos_min": mos_min,
        "mos_max": mos_max,
        "dataset": _dataset_name(config),
        "model_name": config.model.name,
        "backbone": config.model.backbone,
        "media_type": "image" if model_id == "iqa" else "video",
        "checkpoint": str(checkpoint_path),
    }
    logger.info(
        "<green>Loaded {} model: {} | backbone={} | MOS=[{}, {}]</green>",
        model_id,
        config.model.name,
        config.model.backbone,
        mos_min,
        mos_max,
    )


def load_all_models() -> None:
    logger.info("Loading serving models on device: {}", DEVICE)
    model_cache.clear()
    for model_id, checkpoint_path in MODEL_PATHS.items():
        if checkpoint_path.exists():
            try:
                _load_model(model_id, checkpoint_path)
            except Exception as exc:
                logger.exception("Failed to load {} checkpoint: {}", model_id, exc)
        else:
            logger.warning("Checkpoint not found for {}: {}", model_id, checkpoint_path)
    if not model_cache:
        raise RuntimeError("No serving model checkpoints are available")


def save_openapi_schema(application: FastAPI) -> None:
    """Persist the generated OpenAPI document for offline inspection."""
    OPENAPI_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPENAPI_OUTPUT_PATH.write_text(
        json.dumps(application.openapi(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("OpenAPI schema saved: {}", OPENAPI_OUTPUT_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if AUTH_MODE == "api_key" and not API_KEY:
        raise RuntimeError("DEEP_VQA_API_KEY must be set when auth.mode is api_key")
    ensure_runtime_dirs()
    _initialize_frontend_evaluation_store()
    save_openapi_schema(app)
    load_all_models()
    yield


app = FastAPI(
    title="Deep-VQA Quality Assessment API",
    description="REST API for image and video quality assessment inference.",
    version=__version__,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=detail).model_dump(),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(error="Request validation failed", detail=str(exc.errors())).model_dump(),
    )

cors_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"]
    )


def _model_summary(model_id: str, cached: dict[str, Any]) -> ModelSummary:
    return ModelSummary(
        model_id=model_id,
        task_type=str(cached["config"].task_type),
        model_name=cached["model_name"],
        backbone=cached["backbone"],
        dataset=cached["dataset"],
        media_type=cached["media_type"],
        mos_min=cached["mos_min"],
        mos_max=cached["mos_max"],
        checkpoint=cached["checkpoint"],
    )


def _media_size(file_path: str, media_type: str) -> str:
    try:
        if media_type == "video":
            capture = cv2.VideoCapture(file_path)
            if capture.isOpened():
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                capture.release()
                return f"{width}x{height}"
        image = cv2.imread(file_path)
        if image is not None:
            height, width = image.shape[:2]
            return f"{width}x{height}"
    except Exception:
        logger.debug("Unable to read media dimensions: {}", file_path)
    return "unknown"


@app.get(
    f"{API_PREFIX}/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Check service health",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", loaded_models=sorted(model_cache), device=DEVICE)


@app.get(f"{API_PREFIX}/auth/status", response_model=AuthStatusResponse, tags=["auth"])
async def auth_status(request: Request) -> AuthStatusResponse:
    return AuthStatusResponse(required=_auth_required(), authenticated=_is_authenticated(request))


@app.post(f"{API_PREFIX}/auth/login", response_model=AuthStatusResponse, tags=["auth"])
async def auth_login(payload: LoginRequest, request: Request) -> JSONResponse:
    if not _auth_required():
        return JSONResponse(AuthStatusResponse(required=False, authenticated=True).model_dump())
    if not API_KEY or not hmac.compare_digest(payload.api_key, API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    response = JSONResponse(AuthStatusResponse(required=True, authenticated=True).model_dump())
    response.set_cookie(
        AUTH_COOKIE,
        AUTH_TOKEN,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        max_age=86400,
    )
    return response


@app.post(f"{API_PREFIX}/auth/logout", response_model=AuthStatusResponse, tags=["auth"])
async def auth_logout() -> JSONResponse:
    response = JSONResponse(AuthStatusResponse(required=_auth_required(), authenticated=False).model_dump())
    response.delete_cookie(AUTH_COOKIE)
    return response


@app.get(
    f"{API_PREFIX}/models",
    response_model=ModelCollectionResponse,
    tags=["models"],
    summary="List loaded quality assessment models",
)
async def list_models() -> ModelCollectionResponse:
    # Model metadata is intentionally public for the frontend bootstrap.
    items = [_model_summary(model_id, cached) for model_id, cached in sorted(model_cache.items())]
    return ModelCollectionResponse(items=items, count=len(items))


@app.get(
    f"{API_PREFIX}/models/{{model_id}}",
    response_model=ModelSummary,
    responses={404: {"model": ErrorResponse}},
    tags=["models"],
    summary="Get one loaded model",
)
async def get_model(model_id: str) -> ModelSummary:
    cached = model_cache.get(model_id.lower())
    if cached is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")
    return _model_summary(model_id.lower(), cached)


@app.get(
    f"{API_PREFIX}/frontend-evaluations",
    response_model=FrontendEvaluationLogResponse,
    tags=["frontend"],
    summary="List recent frontend evaluation log records",
)
async def list_frontend_evaluations(
    request: Request,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of newest SQLite records to return"),
) -> FrontendEvaluationLogResponse:
    _require_auth(request)
    if EVALUATION_STORE_BACKEND == "none":
        return FrontendEvaluationLogResponse(items=[], count=0)
    items = _read_frontend_evaluation_log(limit, _frontend_session_id(request))
    return FrontendEvaluationLogResponse(items=items, count=len(items))


@app.post(
    f"{API_PREFIX}/subjective-assessments",
    response_model=SubjectiveAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["frontend"],
    summary="Describe image or video quality with Ollama and calculate a Bayesian assessment",
)
async def create_subjective_assessment(
    request: Request,
    file: UploadFile = File(..., description="Image or video file to assess without persistence"),
    video_frames: int = Query(4, ge=1, le=8, description="Evenly sampled frames for video assessment"),
) -> SubjectiveAssessmentResponse:
    """Run an ephemeral Ollama assessment; this route never writes SQLite or JSONL records."""
    _require_auth(request)
    if not file.filename:
        raise HTTPException(status_code=422, detail="Uploaded file must have a filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix in cfg.image_exts:
        media_type = "image"
    elif suffix in cfg.video_exts:
        media_type = "video"
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported media type: {suffix or 'unknown'}")

    tmp_path = ""
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise UploadLimitError(f"Request exceeds the {MAX_UPLOAD_BYTES} byte upload limit")
        tmp_path = await _save_upload_to_temp(file, suffix)
        result = await asyncio.to_thread(
            assess_subjective_quality,
            Path(tmp_path),
            media_type,
            OLLAMA_BASE_URL,
            OLLAMA_QUALITY_MODEL,
            OLLAMA_TIMEOUT_SECONDS,
            video_frames,
        )
        return SubjectiveAssessmentResponse(
            media_type=media_type,
            description=result.description,
            llm_score=result.llm_score,
            bayesian_score=result.bayesian_score,
            credible_interval=result.credible_interval,
            model_used=result.model_used,
            frames_sampled=result.frames_sampled,
        )
    except (UploadLimitError, UploadTimeoutError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except SubjectiveQualityError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        await file.close()


@app.post(
    f"{API_PREFIX}/evaluations",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["evaluations"],
    summary="Evaluate one uploaded image or video",
)
async def create_evaluation(
    request: Request,
    file: UploadFile = File(..., description="Image or video file to evaluate"),
    model_id: str = Query("iqa", description="Serving model role: iqa or vqa"),
) -> EvaluationResponse:
    _require_auth(request)
    model_id = model_id.lower()
    cached = model_cache.get(model_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")
    if not file.filename:
        raise HTTPException(status_code=422, detail="Uploaded file must have a filename")

    suffix = Path(file.filename).suffix.lower()
    expected_exts = cfg.image_exts if cached["media_type"] == "image" else cfg.video_exts
    if suffix not in expected_exts:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{model_id}' accepts {cached['media_type']} files, received '{suffix or 'unknown'}'",
        )

    tmp_path = ""
    started = time.perf_counter()
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise UploadLimitError(f"Request exceeds the {MAX_UPLOAD_BYTES} byte upload limit")
        tmp_path = await _save_upload_to_temp(file, suffix)

        result = predict_single(
            cached["model"],
            Path(tmp_path),
            cached["config"],
            DEVICE,
            cached["mos_min"],
            cached["mos_max"],
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response = EvaluationResponse(
            evaluation_id=f"eval-{time.time_ns()}",
            filename=file.filename,
            model_id=model_id,
            media_type=cached["media_type"],
            task_type=str(cached["config"].task_type),
            model_name=cached["model_name"],
            backbone=cached["backbone"],
            dataset=cached["dataset"],
            score=float(result["raw_score"]),
            mos_score=result.get("mos_score"),
            mos_min=cached["mos_min"],
            mos_max=cached["mos_max"],
            media_size=_media_size(tmp_path, cached["media_type"]),
            latency_ms=latency_ms,
        )
        if request.headers.get("X-Deep-VQA-Client") == "frontend":
            _append_frontend_evaluation_log(response, Path(tmp_path), _frontend_session_id(request))
        return response
    except (UploadLimitError, UploadTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evaluation failed for {}: {}", file.filename, exc)
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        await file.close()


@app.post(
    f"{API_PREFIX}/visualizations",
    response_model=VisualizationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["visualizations"],
    summary="Generate feature maps and Grad-CAM for an image",
)
async def create_visualization(
    request: Request,
    file: UploadFile = File(..., description="Image file to visualize"),
    model_id: str = Query("iqa", description="Loaded image model role"),
) -> VisualizationResponse:
    _require_auth(request)
    if EVALUATION_STORE_BACKEND == "none":
        raise HTTPException(status_code=422, detail="Visualization artifacts are disabled in stateless mode")
    model_id = model_id.lower()
    cached = model_cache.get(model_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")
    if cached["media_type"] != "image":
        raise HTTPException(status_code=422, detail="Feature visualization currently supports image models only")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in cfg.image_exts:
        raise HTTPException(status_code=422, detail="Visualization requires a supported image file")

    tmp_path = ""
    visualization_id = f"viz-{time.time_ns()}"
    # Keep generated visualization artifacts with the IQA report outputs so
    # CLI and browser-generated reports share one durable, inspectable tree.
    output_dir = cfg.resolve(cfg.reports_dir / "iqa-test") / visualization_id
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise UploadLimitError(f"Request exceeds the {MAX_UPLOAD_BYTES} byte upload limit")
        tmp_path = await _save_upload_to_temp(file, suffix)
        image = load_image_tensor(Path(tmp_path), int(cached["config"].model.input_size))
        visualizer = FeatureVisualizer(cached["model"], DEVICE)
        layers = visualizer.default_image_layers()
        result = visualizer.run_image(image, layers=layers, cam_layer="image_backbone")
        feature_maps: dict[str, str] = {}
        for name, activation in result.activations.items():
            path = output_dir / f"{name.replace('.', '_')}.png"
            visualizer.save_feature_grid(activation, path)
            feature_maps[name] = str(path)
        gradcam_path = output_dir / "gradcam.png"
        if result.cam is None:
            raise ValueError("The selected image backbone did not produce a differentiable spatial CAM")
        visualizer.save_cam_overlay(image, result.cam, gradcam_path)
        return VisualizationResponse(
            visualization_id=visualization_id,
            model_id=model_id,
            filename=file.filename or "image",
            score=result.score,
            layers=layers,
            feature_maps=feature_maps,
            gradcam=str(gradcam_path),
        )
    except (UploadLimitError, UploadTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Visualization failed for {}: {}", file.filename, exc)
        raise HTTPException(status_code=500, detail="Visualization failed") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

        await file.close()


def _find_artifact(root: Path, requested_path: str) -> Path | None:
    """Find a requested artifact among files enumerated from the trusted root."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and relative.as_posix() == requested_path:
            return resolved
    return None


@app.get(f"{API_PREFIX}/artifacts/{{artifact_path:path}}", include_in_schema=False)
async def get_artifact(artifact_path: str, request: Request) -> FileResponse:
    """Serve generated visualization artifacts only after API authentication."""
    _require_auth(request)
    if EVALUATION_STORE_BACKEND == "none":
        raise HTTPException(status_code=404, detail="Artifacts are disabled in stateless mode")
    normalized_artifact_path = artifact_path.replace("\\", "/")
    segments = normalized_artifact_path.split("/")
    if (
        not normalized_artifact_path
        or normalized_artifact_path == "."
        or normalized_artifact_path.startswith("/")
        or any(
            not segment
            or segment in {".", ".."}
            or not SAFE_ARTIFACT_SEGMENT_RE.fullmatch(segment)
            for segment in segments
        )
    ):
        raise HTTPException(status_code=404, detail="Artifact not found")

    root = cfg.resolve(cfg.reports_dir / "iqa-test").resolve()
    candidate = _find_artifact(root, normalized_artifact_path)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(candidate)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deploy.api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
