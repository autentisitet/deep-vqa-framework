"""REST API for image and video quality assessment.

The service exposes OpenAPI documentation automatically:

* Swagger UI: ``/docs``
* ReDoc: ``/redoc``
* OpenAPI document: ``/openapi.json``
"""

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

import av
import torch
from fastapi import FastAPI, File, Query, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException

from deploy.config import cfg, ensure_runtime_dirs
from deploy.inference.model_loader import load_checkpoint
from deploy.inference.predictor import PredictionResult, predict_single
from deploy.types import AuthMode
from src import __version__
from src.config.schemas import Config
from src.data.dataset_types import DatasetType, TaskType
from src.data.image_io import image_size_oriented
from src.visualization.feature_visualizer import FeatureVisualizationResult, FeatureVisualizer, load_image_tensor

from .accounts import AccountStore
from .evaluation_store import EvaluationRecord, SQLiteEvaluationStore
from .exceptions import (
    ArtifactNotFoundError,
    AuthenticationRequiredError,
    DeepVQAError,
    ErrorCode,
    FeatureUnavailableError,
    InferenceError,
    InvalidCredentialsError,
    MediaValidationError,
    ModelNotFoundError,
    OriginRejectedError,
    RateLimitError,
    ResourceNotFoundError,
    ValidationError,
    VisualizationError,
)
from .media_upload import (
    StoredUpload,
    UploadLimitError,
    UploadTypeError,
    save_validated_upload,
)
from .schemas import (
    AuthStatusResponse,
    ErrorResponse,
    EvaluationResponse,
    FrontendEvaluationLogResponse,
    FrontendEvaluationRecord,
    FrontendSettingsResponse,
    HealthResponse,
    LoginRequest,
    ModelCollectionResponse,
    ModelSummary,
    RegisterRequest,
    VisualizationResponse,
)

API_PREFIX = "/v1"
DEVICE = cfg.default_device
SAFE_FILE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
MODEL_SETTINGS = cfg.deployment.models
MODEL_PATHS = {model_id: cfg.resolve_path(settings.checkpoint_path) for model_id, settings in MODEL_SETTINGS.items()}
OPENAPI_OUTPUT_PATH = cfg.resolve_path(Path("docs/openapi.json"))
FRONTEND_EVALUATION_DB_PATH = cfg.resolve_path(cfg.endpoints.sqlite_path)
LEGACY_FRONTEND_EVALUATION_LOG_PATH = cfg.resolve_path(cfg.reports_dir / "frontend-evaluations.jsonl")

MAX_UPLOAD_BYTES = cfg.deployment.uploads.max_bytes
UPLOAD_READ_TIMEOUT_SECONDS = cfg.deployment.uploads.read_timeout_seconds
SQLITE_BUSY_TIMEOUT_SECONDS = cfg.deployment.sqlite.busy_timeout_seconds
SQLITE_MAX_DATABASE_BYTES = cfg.deployment.sqlite.max_database_bytes
EVALUATION_STORE_BACKEND = cfg.deployment.evaluation_store.backend
AUTH_MODE = cfg.deployment.auth.mode
API_KEY = os.getenv("DEEP_VQA_API_KEY", "")
AUTH_COOKIE = "deep_vqa_auth"
# Accounts mode uses a separate cookie so switching modes never inherits a
# token from the previous one.
SESSION_COOKIE = "deep_vqa_session"
# Anonymous, server-issued session. It is NOT authentication; it only scopes
# generated artifacts and evaluation history to the browser that created them.
CLIENT_SESSION_COOKIE = "deep_vqa_client"
LOGIN_ATTEMPTS = cfg.deployment.auth.login_attempts
LOGIN_WINDOW_SECONDS = cfg.deployment.auth.login_window_seconds
SESSION_MAX_AGE_SECONDS = cfg.deployment.auth.session_max_age_seconds
ARTIFACT_RETENTION_SECONDS = cfg.deployment.artifacts.retention_seconds
# Bound one sweep so creating an artifact never turns into an unbounded delete.
ARTIFACT_SWEEP_LIMIT = 200
AUTH_TOKEN = hashlib.pbkdf2_hmac(
    "sha256",
    API_KEY.encode(),
    (os.getenv("DEEP_VQA_AUTH_SECRET", "").encode() or secrets.token_bytes(32)),
    310000,
).hex()
LEGACY_SESSION_ID = "legacy"
ACCOUNTS_DB_PATH = cfg.resolve_path(cfg.endpoints.accounts_sqlite_path)
REGISTER_ATTEMPTS = 5
REGISTER_WINDOW_SECONDS = 3600
_frontend_log_lock = threading.Lock()
_evaluation_store = SQLiteEvaluationStore(
    FRONTEND_EVALUATION_DB_PATH, SQLITE_MAX_DATABASE_BYTES, SQLITE_BUSY_TIMEOUT_SECONDS
)
_account_store = AccountStore(ACCOUNTS_DB_PATH, SQLITE_MAX_DATABASE_BYTES, SQLITE_BUSY_TIMEOUT_SECONDS)
_register_attempts: dict[str, deque[float]] = defaultdict(deque)
_register_attempts_lock = threading.Lock()
_RATE_LIMIT_MAX_TRACKED = 4096


def _security_event(
    event: str,
    request: Request,
    **fields: object,
) -> None:
    """Write a small security event without credentials or raw tokens."""
    logger.bind(event=event, client_address=_client_address(request), **fields).info("security event")


def _auth_required() -> bool:
    """Whether the selected deployment mode requires authentication."""
    return AUTH_MODE in {"api_key", "accounts"}


def _session_identity(request: Request) -> tuple[str, str] | None:
    """Return ``(user_id, email)`` for the signed-in account, if any."""
    if AUTH_MODE != "accounts":
        return None
    token = request.cookies.get(SESSION_COOKIE, "").strip()
    if not token:
        return None
    return _account_store.resolve_session(token)


def _is_authenticated(request: Request) -> bool:
    """Whether this request satisfies the configured authentication mode."""
    if not _auth_required():
        return True
    if AUTH_MODE == "accounts":
        return _session_identity(request) is not None
    if not API_KEY:
        return False
    supplied = request.cookies.get(AUTH_COOKIE, "")
    return hmac.compare_digest(supplied, AUTH_TOKEN)


def _client_address(request: Request) -> str:
    """Use the Nginx-provided client address; FastAPI is not public-facing."""
    return request.headers.get("x-real-ip", "").split(",", 1)[0].strip() or (
        request.client.host if request.client else "unknown"
    )


def _prune_rate_limit_table(table: dict[str, deque[float]], now: float, window_seconds: float) -> None:
    """Bound a per-address rate-limit table.

    Callers hold the table's lock. The sweep runs only while the table is over
    the cap, so the common path stays O(1): first drop entries whose window has
    fully expired, then evict the least-recently-active addresses. A login
    flood can therefore never grow the table without bound or make every later
    attempt scan it linearly.
    """
    if len(table) <= _RATE_LIMIT_MAX_TRACKED:
        return
    for address in list(table):
        attempts = table[address]
        if not attempts or now - attempts[-1] >= window_seconds:
            table.pop(address, None)
    if len(table) <= _RATE_LIMIT_MAX_TRACKED:
        return
    overflow = sorted(table, key=lambda address: table[address][-1])[: len(table) - _RATE_LIMIT_MAX_TRACKED]
    for address in overflow:
        table.pop(address, None)


def _check_login_rate_limit(request: Request) -> None:
    """Reject a client address that has exceeded its login-attempt window."""
    now = time.monotonic()
    key = _client_address(request)
    with _login_attempts_lock:
        attempts = _login_attempts[key]
        while attempts and now - attempts[0] >= LOGIN_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= LOGIN_ATTEMPTS:
            retry_after = max(1, int(LOGIN_WINDOW_SECONDS - (now - attempts[0])))
            raise RateLimitError(
                "Too many login attempts; retry later",
                headers={"Retry-After": str(retry_after)},
            )
        attempts.append(now)
        _prune_rate_limit_table(_login_attempts, now, LOGIN_WINDOW_SECONDS)


def _clear_login_rate_limit(request: Request) -> None:
    """Forget login attempts for this address after a successful login."""
    with _login_attempts_lock:
        _login_attempts.pop(_client_address(request), None)


def _check_register_rate_limit(request: Request) -> None:
    """Bound account creation per client address; open signup is the abuse vector."""
    now = time.monotonic()
    key = _client_address(request)
    with _register_attempts_lock:
        attempts = _register_attempts[key]
        while attempts and now - attempts[0] >= REGISTER_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= REGISTER_ATTEMPTS:
            retry_after = max(1, int(REGISTER_WINDOW_SECONDS - (now - attempts[0])))
            raise RateLimitError(
                "Too many registration attempts; retry later",
                headers={"Retry-After": str(retry_after)},
            )
        attempts.append(now)
        _prune_rate_limit_table(_register_attempts, now, REGISTER_WINDOW_SECONDS)


def _clear_register_rate_limit(request: Request) -> None:
    """Forget registration attempts for this address after a successful signup."""
    with _register_attempts_lock:
        _register_attempts.pop(_client_address(request), None)


def _set_session_cookie(
    response: JSONResponse,
    request: Request,
    token: str,
) -> None:
    """Attach the accounts session cookie to a response."""
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=_cookie_secure(request),
        samesite=_cookie_samesite(request),
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
    )


def _request_origin(request: Request) -> str:
    """Origin this request reached the API through, honouring the proxy header."""
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    scheme = forwarded_proto or request.url.scheme
    return f"{scheme}://{request.headers.get('host', request.url.netloc)}".rstrip("/")


def _allowed_origins(request: Request) -> set[str]:
    """Only the browser-facing origin is trusted for cookie state changes."""
    return {_request_origin(request)}


def _check_csrf_origin(request: Request) -> None:
    """Reject cross-origin cookie-authenticated state changes when Origin exists."""
    if not _auth_required() or request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    origin = request.headers.get("origin", "").strip()
    if not origin:
        return
    if origin.rstrip("/") not in _allowed_origins(request):
        raise OriginRejectedError("Cross-origin request rejected")


def _require_auth(request: Request) -> None:
    """Enforce CSRF origin and authentication, raising 401 when unauthenticated."""
    _check_csrf_origin(request)
    if not _is_authenticated(request):
        raise AuthenticationRequiredError("Authentication required")


class LoadedModel(TypedDict):
    model: torch.nn.Module
    config: Config
    mos_min: float
    mos_max: float
    dataset: str
    model_name: str
    backbone: str
    media_type: DatasetType
    checkpoint: str


model_cache: dict[str, LoadedModel] = {}
_heavy_task_lock = asyncio.Lock()
_login_attempts: dict[str, deque[float]] = defaultdict(deque)
_login_attempts_lock = threading.Lock()


def _mos_range(config: Config) -> tuple[float, float]:
    """Return the validated MOS range declared by a checkpoint config."""
    mos_min = float(config.dataset.mos_min)
    mos_max = float(config.dataset.mos_max)
    if mos_max <= mos_min:
        raise ValueError(f"Invalid checkpoint MOS range: [{mos_min}, {mos_max}]")
    return mos_min, mos_max


def _dataset_name(config: Config) -> str:
    """Return the dataset display name from a checkpoint config."""
    dataset = config.dataset
    return str(getattr(dataset, "name", None) or getattr(dataset, "registry_key", "unknown"))


def _cookie_secure(request: Request) -> bool:
    """Keep cookies transport-secure when the request arrived over HTTPS."""
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    return request.url.scheme == "https" or forwarded == "https"


def _cookie_samesite(request: Request) -> Literal["strict", "none"]:
    """Same-origin cookies stay Strict on HTTP and HTTPS deployments."""
    return "strict"


def _client_session_owner(request: Request) -> str | None:
    """Return a non-reversible owner key for the caller's anonymous session.

    The cookie carries a random token; only its digest is persisted or compared,
    so a leaked artifact directory never exposes the session token itself.
    """
    token = request.cookies.get(CLIENT_SESSION_COOKIE, "").strip()
    if not token:
        return None
    return hashlib.sha256(token.encode()).hexdigest()


def _ensure_client_session(request: Request, response: Response) -> str:
    """Return the caller's session owner key, issuing a session when absent.

    This scopes artifacts to a browser; it is not authentication. It ensures one
    visitor cannot read another visitor's generated artifacts or history even
    when both share the same deployment API key.
    """
    owner = _client_session_owner(request)
    if owner is not None:
        return owner
    token = secrets.token_urlsafe(32)
    response.set_cookie(
        CLIENT_SESSION_COOKIE,
        token,
        httponly=True,
        secure=_cookie_secure(request),
        samesite=_cookie_samesite(request),
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
    )
    return hashlib.sha256(token.encode()).hexdigest()


def _current_owner(request: Request) -> str | None:
    """Owner key for artifacts and history.

    In accounts mode this is the account id, so the data survives clearing
    cookies and follows the person across devices. Otherwise it is the digest of
    the anonymous browser session.
    """
    identity = _session_identity(request)
    if identity is not None:
        return identity[0]
    return _client_session_owner(request)


def _ensure_owner(request: Request, response: Response) -> str:
    """Return the caller's owner key, issuing an anonymous session if needed."""
    identity = _session_identity(request)
    if identity is not None:
        return identity[0]
    return _ensure_client_session(request, response)


def _write_artifact_owner(directory: Path, owner: str) -> None:
    """Record which session produced an artifact directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".owner").write_text(owner, encoding="utf-8")


def _artifact_owner(directory: Path) -> str | None:
    """Read the recorded owner for an artifact directory, if present."""
    try:
        return (directory / ".owner").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _generated_file_root() -> Path:
    """Directory that holds generated visualization artifacts."""
    return cfg.resolve_path(cfg.reports_dir / "iqa-test").resolve()


def _purge_expired_artifacts(root: Path, now: float | None = None) -> int:
    """Delete generated visualization directories past their retention window.

    Only ``viz-*`` directories are considered, so CLI report output and
    unrelated files under the same root are never touched. This runs when a new
    artifact is created, not during startup, because deleting data on boot would
    make a restart destructive.
    """
    if not root.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - ARTIFACT_RETENTION_SECONDS

    expired: list[tuple[float, str, Path]] = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir() or not entry.name.startswith("viz-"):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime <= cutoff:
            expired.append((mtime, entry.name, entry))

    expired.sort()
    removed = 0
    for _, _, entry in expired[:ARTIFACT_SWEEP_LIMIT]:
        try:
            shutil.rmtree(entry)
        except OSError as exc:
            logger.warning("Could not remove expired artifact {}: {}", entry, exc)
            continue
        removed += 1
    return removed


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of an uploaded media file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _save_upload_to_temp(file: UploadFile) -> StoredUpload:
    """Store a bounded upload after validating its file-header magic."""
    return await save_validated_upload(
        file,
        MAX_UPLOAD_BYTES,
        UPLOAD_READ_TIMEOUT_SECONDS,
        cfg.endpoints.upload_dir,
    )


def _safe_client_filename(filename: str | None, fallback: str) -> str:
    """Return display-only basename; it is never used as a filesystem path."""
    basename = Path((filename or "").replace("\\", "/")).name
    cleaned = "".join(
        character for character in basename if character.isprintable() and character not in {"\x00", "\r", "\n"}
    )
    return cleaned[:255] or fallback


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


def _read_frontend_evaluation_log(limit: int, session_id: str | None = None) -> list[dict[str, object]]:
    """Return the newest frontend evaluation records from SQLite."""
    if EVALUATION_STORE_BACKEND == "none":
        return []
    _evaluation_store.path = FRONTEND_EVALUATION_DB_PATH
    _evaluation_store.initialize(LEGACY_FRONTEND_EVALUATION_LOG_PATH)
    return _evaluation_store.read(limit, session_id)


def _load_model(model_id: str, checkpoint_path: Path) -> LoadedModel:
    """Load one checkpoint and reject it when its metadata drifts from the deployment profile."""
    model, config = load_checkpoint(checkpoint_path, device=DEVICE)
    mos_min, mos_max = _mos_range(config)
    settings = MODEL_SETTINGS[model_id]
    actual: dict[str, str | float] = {
        "task_type": str(config.task_type),
        "media_type": "image" if str(config.task_type) == "iqa" else "video",
        "model_name": str(config.model.name),
        "backbone": str(config.model.backbone),
        "dataset": _dataset_name(config),
        "mos_min": mos_min,
        "mos_max": mos_max,
    }
    expected: dict[str, str | float] = {
        "task_type": settings.task_type,
        "media_type": settings.media_type,
        "model_name": settings.model_name,
        "backbone": settings.backbone,
        "dataset": settings.dataset,
        "mos_min": settings.mos_min,
        "mos_max": settings.mos_max,
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if actual[name] != expected_value
        and not (name in {"mos_min", "mos_max"} and abs(float(actual[name]) - float(expected_value)) <= 1e-6)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: checkpoint={actual[name]!r}, configuration={expected[name]!r}" for name in mismatches
        )
        raise ValueError(f"Checkpoint metadata does not match deployment configuration for '{model_id}': {details}")
    cached: LoadedModel = {
        "model": model,
        "config": config,
        "mos_min": mos_min,
        "mos_max": mos_max,
        "dataset": _dataset_name(config),
        "model_name": config.model.name,
        "backbone": config.model.backbone,
        "media_type": settings.media_type,
        "checkpoint": str(checkpoint_path),
    }
    logger.info(
        "<green>Loaded {} model into memory: {} | backbone={} | MOS=[{}, {}]</green>",
        model_id,
        config.model.name,
        config.model.backbone,
        mos_min,
        mos_max,
    )
    return cached


def load_all_models() -> None:
    """Load each configured checkpoint once during worker startup."""
    model_cache.clear()
    for model_id, checkpoint_path in MODEL_PATHS.items():
        model_cache[model_id] = _load_model(model_id, checkpoint_path)
    if set(model_cache) != set(MODEL_PATHS):
        raise RuntimeError("Not all configured serving models were loaded")


def _loaded_model(model_id: str) -> LoadedModel:
    """Return a model loaded during startup."""
    try:
        return model_cache[model_id]
    except KeyError as exc:
        raise RuntimeError(f"Model '{model_id}' was not loaded during startup") from exc


@asynccontextmanager
async def _heavy_operation():
    """Serialize memory-heavy work when enabled in the selected mode."""
    if cfg.deployment.inference_runtime.serialize_heavy_requests:
        async with _heavy_task_lock:
            yield
        return
    yield


def validate_model_catalog() -> None:
    """Fail startup when no configured serving checkpoint is available."""
    missing = [str(path) for path in MODEL_PATHS.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Serving model checkpoints are missing: {', '.join(missing)}")


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
    """Validate configuration, initialize stores, persist OpenAPI, and preload models."""
    if AUTH_MODE == "api_key" and not API_KEY:
        raise RuntimeError("DEEP_VQA_API_KEY must be set when auth.mode is api_key")
    if AUTH_MODE == "accounts":
        # Creating tables is not destructive; expired sessions are purged when
        # someone signs in, never during startup.
        _account_store.initialize()
    ensure_runtime_dirs()
    _initialize_frontend_evaluation_store()
    save_openapi_schema(app)
    validate_model_catalog()
    if not cfg.deployment.inference_runtime.preload_models:
        raise RuntimeError("Deployment configuration must load models during startup")
    load_all_models()
    yield
    model_cache.clear()


app = FastAPI(
    title="Deep-VQA Quality Assessment API",
    description="REST API for image and video quality assessment inference.",
    version=__version__,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    redirect_slashes=False,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Convert ``HTTPException`` into the standard ``ErrorResponse`` shape."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    codes = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.METHOD_NOT_ALLOWED}
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=codes.get(exc.status_code, ErrorCode.HTTP_ERROR), error=detail).model_dump(),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert request validation failures into a 422 ``ErrorResponse``."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code=ErrorCode.VALIDATION_ERROR,
            error="Request validation failed",
            detail="; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(DeepVQAError)
async def deepvqa_error_handler(_: Request, exc: DeepVQAError) -> JSONResponse:
    """Map typed domain errors to a stable HTTP status and message."""
    if exc.http_status_code >= 500:
        logger.opt(exception=exc).error("Service error [{}]: {}", exc.code, exc)
    return JSONResponse(
        status_code=exc.http_status_code,
        content=ErrorResponse(code=exc.code, error=str(exc)).model_dump(),
        headers=(
            {**(exc.headers or {}), "WWW-Authenticate": 'Session realm="Deep-VQA"'}
            if exc.http_status_code == 401
            else exc.headers
        ),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Keep unexpected failures in logs and return a safe public response."""
    logger.opt(exception=exc).error("Unexpected service error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(code=ErrorCode.INTERNAL_ERROR, error="Internal server error").model_dump(),
    )


def _model_summary(model_id: str) -> ModelSummary:
    """Build the public model metadata summary for one configured model."""
    settings = MODEL_SETTINGS[model_id]
    return ModelSummary(
        model_id=model_id,
        loaded=model_id in model_cache,
        task_type=settings.task_type,
        model_name=settings.model_name,
        backbone=settings.backbone,
        dataset=settings.dataset,
        media_type=settings.media_type,
        mos_min=settings.mos_min,
        mos_max=settings.mos_max,
        checkpoint=Path(settings.checkpoint_path).name,
    )


def _media_size(file_path: str, media_type: str) -> str:
    """Read media pixel dimensions as ``WxH``, or ``"unknown"`` on failure."""
    try:
        if media_type == "video":
            with av.open(file_path) as container:
                stream = container.streams.video[0]
                return f"{stream.codec_context.width}x{stream.codec_context.height}"
        width, height = image_size_oriented(Path(file_path))
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
    """Report core IQA/VQA readiness."""
    return HealthResponse(
        status="ok",
        loaded_models=sorted(model_cache),
        device=DEVICE,
    )


@app.get(f"{API_PREFIX}/auth/status", response_model=AuthStatusResponse, tags=["auth"])
async def auth_status(request: Request) -> JSONResponse:
    """Report whether authentication is required and whether this request is authenticated."""
    identity = _session_identity(request)
    return JSONResponse(
        AuthStatusResponse(
            required=_auth_required(),
            authenticated=_is_authenticated(request),
            mode=AUTH_MODE,
            email=identity[1] if identity else None,
        ).model_dump(),
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    f"{API_PREFIX}/frontend-settings",
    response_model=FrontendSettingsResponse,
    tags=["frontend"],
    summary="Report which frontend features are enabled",
)
async def frontend_settings(request: Request) -> FrontendSettingsResponse:
    """Report the feature flags the frontend should enable for this visitor."""
    persistence_enabled = EVALUATION_STORE_BACKEND == "sqlite"
    identity = _session_identity(request)
    return FrontendSettingsResponse(
        auth_required=_auth_required(),
        authenticated=_is_authenticated(request),
        auth_mode=AUTH_MODE,
        email=identity[1] if identity else None,
        evaluation_history_enabled=persistence_enabled,
        visualizations_enabled=persistence_enabled,
    )


@app.post(
    f"{API_PREFIX}/auth/register",
    response_model=AuthStatusResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["auth"],
    summary="Create an account (accounts authentication mode only)",
)
async def auth_register(payload: RegisterRequest, request: Request) -> JSONResponse:
    """Create an account and sign the caller in (accounts authentication mode only)."""
    if AUTH_MODE != "accounts":
        raise ResourceNotFoundError("Registration is only available when auth.mode is accounts")
    _check_register_rate_limit(request)
    user_id, email = _account_store.register(payload.email, payload.password)
    _clear_register_rate_limit(request)
    _security_event("account.registered", request, email=email)
    token = _account_store.create_session(user_id, SESSION_MAX_AGE_SECONDS)
    response = JSONResponse(
        AuthStatusResponse(
            required=True,
            authenticated=True,
            mode=AuthMode.ACCOUNTS,
            email=email,
        ).model_dump(),
        status_code=status.HTTP_201_CREATED,
        headers={"Cache-Control": "no-store"},
    )
    _set_session_cookie(response, request, token)
    return response


@app.post(
    f"{API_PREFIX}/auth/login",
    response_model=AuthStatusResponse,
    responses={429: {"model": ErrorResponse}},
    tags=["auth"],
)
async def auth_login(payload: LoginRequest, request: Request) -> JSONResponse:
    """Sign the caller in with account credentials or an API key and set the auth cookie."""
    if not _auth_required():
        return JSONResponse(
            AuthStatusResponse(
                required=False,
                authenticated=True,
                mode=AUTH_MODE,
            ).model_dump(),
            headers={"Cache-Control": "no-store"},
        )
    _check_login_rate_limit(request)
    if AUTH_MODE == "accounts":
        if not payload.email or not payload.password:
            raise ValidationError("email and password are required")
        identity = _account_store.authenticate(payload.email, payload.password)
        if identity is None:
            # Identical response for an unknown email and a wrong password so the
            # endpoint cannot be used to enumerate registered accounts.
            _security_event("auth.login_failed", request)
            raise InvalidCredentialsError("Invalid email or password")
        user_id, email = identity
        _clear_login_rate_limit(request)
        _security_event("auth.login_succeeded", request, email=email)
        _account_store.purge_expired_sessions()
        token = _account_store.create_session(user_id, SESSION_MAX_AGE_SECONDS)
        response = JSONResponse(
            AuthStatusResponse(
                required=True,
                authenticated=True,
                mode=AuthMode.ACCOUNTS,
                email=email,
            ).model_dump(),
            headers={"Cache-Control": "no-store"},
        )
        _set_session_cookie(response, request, token)
        return response
    if not API_KEY or not payload.api_key or not hmac.compare_digest(payload.api_key, API_KEY):
        _security_event("auth.api_key_login_failed", request)
        raise InvalidCredentialsError("Invalid API key")
    _clear_login_rate_limit(request)
    _security_event("auth.api_key_login_succeeded", request)
    response = JSONResponse(
        AuthStatusResponse(
            required=True,
            authenticated=True,
            mode=AuthMode.API_KEY,
        ).model_dump(),
        headers={"Cache-Control": "no-store"},
    )
    response.set_cookie(
        AUTH_COOKIE,
        AUTH_TOKEN,
        httponly=True,
        secure=_cookie_secure(request),
        samesite=_cookie_samesite(request),
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
    )
    return response


@app.post(f"{API_PREFIX}/auth/logout", response_model=AuthStatusResponse, tags=["auth"])
async def auth_logout(request: Request) -> JSONResponse:
    """End the current authentication and clear the auth and session cookies."""
    _require_auth(request)
    if AUTH_MODE == "accounts":
        _account_store.delete_session(request.cookies.get(SESSION_COOKIE, "").strip())
    _security_event("auth.logout", request)
    response = JSONResponse(
        AuthStatusResponse(required=_auth_required(), authenticated=False, mode=AUTH_MODE).model_dump(),
        headers={"Cache-Control": "no-store"},
    )
    response.delete_cookie(AUTH_COOKIE, path="/")
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get(
    f"{API_PREFIX}/models",
    response_model=ModelCollectionResponse,
    tags=["models"],
    summary="List configured quality assessment models",
)
async def list_models() -> ModelCollectionResponse:
    """Return metadata for every configured quality assessment model."""
    # Model metadata is intentionally public for the frontend bootstrap.
    items = [_model_summary(model_id) for model_id in sorted(MODEL_SETTINGS)]
    return ModelCollectionResponse(items=items, count=len(items))


@app.get(
    f"{API_PREFIX}/models/{{model_id}}",
    response_model=ModelSummary,
    responses={404: {"model": ErrorResponse}},
    tags=["models"],
    summary="Get one configured quality assessment model",
)
async def get_model(model_id: str) -> ModelSummary:
    """Return metadata for one configured model, or 404 when unknown."""
    normalized = model_id.lower()
    if normalized not in MODEL_SETTINGS:
        raise ModelNotFoundError(f"Model '{model_id}' is not configured")
    return _model_summary(normalized)


@app.get(
    f"{API_PREFIX}/frontend-evaluations",
    response_model=FrontendEvaluationLogResponse,
    tags=["frontend"],
    summary="List recent frontend evaluation log records",
)
async def list_frontend_evaluations(
    request: Request,
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Maximum number of newest SQLite records to return",
    ),
) -> FrontendEvaluationLogResponse:
    """Return the newest evaluation records owned by the caller's session."""
    _require_auth(request)
    if EVALUATION_STORE_BACKEND == "none":
        return FrontendEvaluationLogResponse(items=[], count=0)
    owner = _current_owner(request)
    if owner is None:
        # No account and no anonymous session: nothing this visitor may claim.
        return FrontendEvaluationLogResponse(items=[], count=0)
    items = _read_frontend_evaluation_log(limit, owner)
    return FrontendEvaluationLogResponse(
        items=[FrontendEvaluationRecord.model_validate(item) for item in items], count=len(items)
    )


def _check_upload_length(content_length: str | None) -> None:
    """Reject a declared ``Content-Length`` above the upload limit before reading."""
    if content_length and int(content_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
        raise UploadLimitError(f"Request exceeds the {MAX_UPLOAD_BYTES} byte upload limit")


def _reject_media_mismatch(stored: StoredUpload, expected: str, model_id: str) -> None:
    """Reject an upload whose detected media type does not match the model."""
    if stored.media.media_type != expected:
        raise UploadTypeError(f"Model '{model_id}' accepts {expected} media, detected {stored.media.media_type}")


def _require_image_media(stored: StoredUpload) -> None:
    """Reject an upload that is not an image for visualization."""
    if stored.media.media_type != "image":
        raise UploadTypeError("Visualization requires image file magic")


def _evaluation_response(
    file: UploadFile,
    stored: StoredUpload,
    model_id: str,
    cached: LoadedModel,
    result: PredictionResult,
    latency_ms: float,
) -> EvaluationResponse:
    """Build the evaluation response from inference results and measured latency."""
    return EvaluationResponse(
        evaluation_id=f"eval-{time.time_ns()}",
        filename=_safe_client_filename(file.filename, stored.path.name),
        model_id=model_id,
        media_type=cached["media_type"],
        task_type=TaskType(cached["config"].task_type),
        model_name=cached["model_name"],
        backbone=cached["backbone"],
        dataset=cached["dataset"],
        score=float(result["raw_score"]),
        mos_score=result.get("mos_score"),
        mos_min=cached["mos_min"],
        mos_max=cached["mos_max"],
        media_size=_media_size(str(stored.path), cached["media_type"]),
        latency_ms=latency_ms,
    )


def _write_feature_maps(
    visualizer: FeatureVisualizer, result: FeatureVisualizationResult, output_dir: Path
) -> dict[str, str]:
    """Persist each activation grid and return the layer-name to file mapping."""
    feature_maps: dict[str, str] = {}
    for name, activation in result.activations.items():
        path = output_dir / f"{name.replace('.', '_')}.png"
        visualizer.save_feature_grid(activation, path)
        feature_maps[name] = str(path)
    return feature_maps


def _save_gradcam(
    visualizer: FeatureVisualizer, image: torch.Tensor, result: FeatureVisualizationResult, output_dir: Path
) -> Path:
    """Persist the Grad-CAM overlay, raising when no spatial CAM was produced."""
    if result.cam is None:
        raise FeatureUnavailableError("The selected image backbone did not produce a differentiable spatial CAM")
    path = output_dir / "gradcam.png"
    visualizer.save_cam_overlay(image, result.cam, path)
    return path


@app.post(
    f"{API_PREFIX}/evaluations",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["evaluations"],
    summary="Evaluate one uploaded image or video",
)
async def create_evaluation(
    request: Request,
    http_response: Response,
    file: UploadFile = File(..., description="Image or video file to evaluate"),
    model_id: str = Query("iqa", description="Model identifier: iqa or vqa"),
) -> EvaluationResponse:
    """Evaluate one uploaded image or video with the selected quality model.

    The upload is validated and persisted before the model is touched, then
    inference runs inside the heavy-request serialization policy. On success
    the measured scores and latency are returned; frontend-tagged requests are
    also appended to the evaluation history store.
    """
    _require_auth(request)
    session_owner = _ensure_owner(request, http_response)
    model_id = model_id.lower()
    settings = MODEL_SETTINGS.get(model_id)
    if settings is None:
        raise ModelNotFoundError(f"Model '{model_id}' is not configured")
    if not file.filename:
        raise MediaValidationError("Uploaded file must have a filename")

    tmp_path = ""
    started = time.perf_counter()
    try:
        # 1. Validate and persist the upload before touching the model.
        _check_upload_length(request.headers.get("content-length"))
        stored = await _save_upload_to_temp(file)
        tmp_path = str(stored.path)
        _reject_media_mismatch(stored, settings.media_type, model_id)

        # 2. Run inference inside the heavy-request serialization lock.
        async with _heavy_operation():
            cached = _loaded_model(model_id)
            result = await asyncio.to_thread(
                predict_single,
                cached["model"],
                Path(tmp_path),
                cached["config"],
                DEVICE,
                cached["mos_min"],
                cached["mos_max"],
            )
        # 3. Build the response from the measured values.
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response = _evaluation_response(
            file=file,
            stored=stored,
            model_id=model_id,
            cached=cached,
            result=result,
            latency_ms=latency_ms,
        )
        # 4. Record history for the frontend when persistence is enabled.
        if request.headers.get("X-Deep-VQA-Client") == "frontend" and EVALUATION_STORE_BACKEND != "none":
            _append_frontend_evaluation_log(response, Path(tmp_path), session_owner)
        return response
    except DeepVQAError:
        raise
    except ValueError as exc:
        raise MediaValidationError(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise InferenceError("Inference failed") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        await file.close()


@app.post(
    f"{API_PREFIX}/visualizations",
    response_model=VisualizationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["visualizations"],
    summary="Generate feature maps and Grad-CAM for an image",
)
async def create_visualization(
    request: Request,
    http_response: Response,
    file: UploadFile = File(..., description="Image file to visualize"),
    model_id: str = Query("iqa", description="Image model identifier"),
) -> VisualizationResponse:
    """Generate feature maps and a Grad-CAM overlay for an uploaded image.

    The produced PNG files are written under an unguessable directory, tagged
    with the caller's session owner, and only served back to that owner. The
    image backbone must produce a differentiable spatial CAM for Grad-CAM.
    """
    _require_auth(request)
    if EVALUATION_STORE_BACKEND == "none":
        raise FeatureUnavailableError("This mode does not save visualization files")
    session_owner = _ensure_owner(request, http_response)
    model_id = model_id.lower()
    settings = MODEL_SETTINGS.get(model_id)
    if settings is None:
        raise ModelNotFoundError(f"Model '{model_id}' is not configured")
    if settings.media_type != "image":
        raise FeatureUnavailableError("Feature visualization currently supports image models only")
    tmp_path = ""
    # An unguessable id keeps the artifact URL from being derived from its
    # creation time; ownership is still enforced by _artifact_owner().
    visualization_id = f"viz-{secrets.token_urlsafe(16)}"
    # Keep generated visualization files with the IQA report outputs so the
    # CLI and browser use the same directory.
    output_dir = _generated_file_root() / visualization_id
    try:
        # 1. Validate and persist the upload before touching the model.
        _check_upload_length(request.headers.get("content-length"))
        stored = await _save_upload_to_temp(file)
        tmp_path = str(stored.path)
        _require_image_media(stored)
        # 2. Generate feature maps and Grad-CAM, writing PNG files to the output directory.
        async with _heavy_operation():
            cached = _loaded_model(model_id)
            image = load_image_tensor(Path(tmp_path), int(cached["config"].model.input_size))
            visualizer = FeatureVisualizer(cached["model"], DEVICE)
            layers = visualizer.default_image_layers()
            result = visualizer.run_image(image, layers=layers, cam_layer="image_backbone")
            feature_maps = _write_feature_maps(visualizer, result, output_dir)
            gradcam_path = _save_gradcam(visualizer, image, result, output_dir)
        # 3. Record artifact ownership and sweep expired artifacts, then respond.
        _write_artifact_owner(output_dir, session_owner)
        _purge_expired_artifacts(_generated_file_root())
        return VisualizationResponse(
            visualization_id=visualization_id,
            model_id=model_id,
            filename=_safe_client_filename(file.filename, stored.path.name),
            score=result.score,
            layers=layers,
            feature_maps=feature_maps,
            gradcam=str(gradcam_path),
        )
    except DeepVQAError:
        raise
    except ValueError as exc:
        raise MediaValidationError(str(exc)) from exc
    except Exception as exc:
        raise VisualizationError("Visualization failed") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

        await file.close()


def _find_generated_file(root: Path, requested_path: str) -> Path | None:
    """Resolve one generated file while rejecting paths outside the report directory."""
    try:
        candidate = (root / requested_path).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


@app.get(f"{API_PREFIX}/files/{{file_path:path}}", include_in_schema=False)
async def get_generated_file(file_path: str, request: Request) -> FileResponse:
    """Return a generated visualization file owned by the caller's session.

    Authentication only proves the caller may use the service. Because every
    visitor shares one deployment API key, authentication alone would let any
    visitor read another visitor's artifacts; the anonymous session recorded
    next to each artifact is what actually isolates them.
    """
    _require_auth(request)
    if EVALUATION_STORE_BACKEND == "none":
        raise ResourceNotFoundError("This mode does not save visualization files")
    owner = _current_owner(request)
    if owner is None:
        raise ArtifactNotFoundError("Generated file not found")
    normalized_file_path = file_path.replace("\\", "/")
    segments = normalized_file_path.split("/")
    if (
        not normalized_file_path
        or normalized_file_path == "."
        or normalized_file_path.startswith("/")
        or any(
            not segment
            or segment.startswith(".")
            or segment in {".", ".."}
            or not SAFE_FILE_SEGMENT_RE.fullmatch(segment)
            for segment in segments
        )
    ):
        raise ArtifactNotFoundError("Generated file not found")

    root = _generated_file_root()
    recorded_owner = _artifact_owner(root / segments[0])
    # Return 404 rather than 403 so the response does not confirm that another
    # session's artifact exists.
    if recorded_owner is None or not hmac.compare_digest(recorded_owner, owner):
        raise ArtifactNotFoundError("Generated file not found")
    candidate = _find_generated_file(root, normalized_file_path)
    if candidate is None:
        raise ArtifactNotFoundError("Generated file not found")
    return FileResponse(candidate)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "deploy.web.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
