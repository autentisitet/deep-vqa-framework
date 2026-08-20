"""REST API for image and video quality assessment.

The service exposes OpenAPI documentation automatically:

* Swagger UI: ``/docs``
* ReDoc: ``/redoc``
* OpenAPI document: ``/openapi.json``
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from deploy.core.inference import predict_single
from deploy.core.model_loader import load_checkpoint
from deploy.core.runtime_config import cfg, ensure_runtime_dirs
from src.visualization.feature_visualizer import FeatureVisualizer, load_image_tensor


API_PREFIX = "/v1"
DEVICE = cfg.default_device
MODEL_PATHS = {
    "iqa": cfg.resolve(cfg.iqa_model_path),
    "vqa": cfg.resolve(cfg.vqa_model_path),
}
OPENAPI_OUTPUT_PATH = cfg.resolve(Path("docs/openapi.json"))
FRONTEND_EVALUATION_LOG_PATH = cfg.resolve(cfg.reports_dir / "frontend-evaluations.jsonl")
_frontend_log_lock = threading.Lock()


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


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of an uploaded media file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_frontend_evaluation_log(response: EvaluationResponse, media_path: Path) -> None:
    """Append one successful browser evaluation as a JSON Lines record."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_name": response.filename,
        "file_hash": _sha256(media_path),
        "task_type": response.task_type,
        "model_used": response.model_name,
        "mos_score": response.mos_score,
        "mos_interval": [response.mos_min, response.mos_max],
        "inference_time_ms": response.latency_ms,
    }
    FRONTEND_EVALUATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _frontend_log_lock, FRONTEND_EVALUATION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    ensure_runtime_dirs()
    save_openapi_schema(app)
    load_all_models()
    yield


app = FastAPI(
    title="Deep-VQA Quality Assessment API",
    description="REST API for image and video quality assessment inference.",
    version="5.0.0",
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


@app.get(
    f"{API_PREFIX}/models",
    response_model=ModelCollectionResponse,
    tags=["models"],
    summary="List loaded quality assessment models",
)
async def list_models() -> ModelCollectionResponse:
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

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    started = time.perf_counter()
    try:
        with open(tmp_fd, "wb") as target:
            shutil.copyfileobj(file.file, target)

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
            _append_frontend_evaluation_log(response, Path(tmp_path))
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evaluation failed for {}: {}", file.filename, exc)
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    finally:
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
    file: UploadFile = File(..., description="Image file to visualize"),
    model_id: str = Query("iqa", description="Loaded image model role"),
) -> VisualizationResponse:
    model_id = model_id.lower()
    cached = model_cache.get(model_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")
    if cached["media_type"] != "image":
        raise HTTPException(status_code=422, detail="Feature visualization currently supports image models only")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in cfg.image_exts:
        raise HTTPException(status_code=422, detail="Visualization requires a supported image file")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    visualization_id = f"viz-{time.time_ns()}"
    output_dir = cfg.resolve(Path("results/diagnostics")) / visualization_id
    try:
        with open(tmp_fd, "wb") as target:
            shutil.copyfileobj(file.file, target)
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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Visualization failed for {}: {}", file.filename, exc)
        raise HTTPException(status_code=500, detail="Visualization failed") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deploy.api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
