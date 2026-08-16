#!/usr/bin/env python
# -*- coding: utf-8 -*-
# deploy/api.py
"""
FastAPI inference service.

Usage:
    uv run python -m deploy.api
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from deploy.core.runtime_config import cfg, ensure_runtime_dirs
from deploy.core.model_loader import load_checkpoint
from deploy.core.inference import (
    denormalize,
    predict_single,
)
from src.config.schemas import Config


# ---------- Configuration ----------
DEFAULT_IQA_MODEL_PATH = cfg.iqa_model_path
DEFAULT_VQA_MODEL_PATH = cfg.vqa_model_path
DEVICE = cfg.default_device


# ---------- FastAPI ----------
app = FastAPI(title="Deep-VQA Unified MOS API", version="4.0.0")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

model_cache: Dict[str, Dict[str, Any]] = {}


def get_mos_params(config: Config) -> tuple[float, float]:
    """Read and validate the MOS range embedded in the checkpoint config."""
    mos_min = float(config.dataset.mos_min)
    mos_max = float(config.dataset.mos_max)
    if mos_max <= mos_min:
        raise ValueError(f"Invalid checkpoint MOS range: [{mos_min}, {mos_max}]")
    return mos_min, mos_max


def load_all_models():
    logger.info(f"[INFO] Using device: {DEVICE}")

    if DEFAULT_IQA_MODEL_PATH.exists():
        logger.info(f"[INFO] Loading IQA model: {DEFAULT_IQA_MODEL_PATH}")
        model, config_obj = load_checkpoint(DEFAULT_IQA_MODEL_PATH, device=DEVICE)
        mos_min, mos_max = get_mos_params(config_obj)
        dataset_cfg = getattr(config_obj, "dataset", {}) or {}
        if hasattr(dataset_cfg, "model_dump"):
            dataset_cfg = dataset_cfg.model_dump()
        model_cache["iqa"] = {
            "model": model,
            "config": config_obj,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "dataset": dataset_cfg.get("name", dataset_cfg.get("registry_key", "unknown")),
            "model_name": config_obj.model.name,
            "backbone": config_obj.model.backbone,
        }
        logger.info(f"[OK] IQA loaded (MOS: {mos_min:.3f}~{mos_max:.3f})")
    else:
        logger.warning(f"[WARN] IQA model not found: {DEFAULT_IQA_MODEL_PATH}")

    if DEFAULT_VQA_MODEL_PATH.exists():
        logger.info(f"[INFO] Loading VQA model: {DEFAULT_VQA_MODEL_PATH}")
        model, config_obj = load_checkpoint(DEFAULT_VQA_MODEL_PATH, device=DEVICE)
        mos_min, mos_max = get_mos_params(config_obj)
        dataset_cfg = getattr(config_obj, "dataset", {}) or {}
        if hasattr(dataset_cfg, "model_dump"):
            dataset_cfg = dataset_cfg.model_dump()
        model_cache["vqa"] = {
            "model": model,
            "config": config_obj,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "dataset": dataset_cfg.get("name", dataset_cfg.get("registry_key", "unknown")),
            "model_name": config_obj.model.name,
            "backbone": config_obj.model.backbone,
        }
        logger.info(f"[OK] VQA loaded (MOS: {mos_min:.3f}~{mos_max:.3f})")
    else:
        logger.warning(f"[WARN] VQA model not found: {DEFAULT_VQA_MODEL_PATH}")

    if not model_cache:
        raise RuntimeError("No models loaded")


@app.on_event("startup")
async def startup_event():
    ensure_runtime_dirs()
    load_all_models()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": list(model_cache.keys()),
        "device": DEVICE,
    }


def get_media_size(file_path: str, media_type: str) -> str:
    try:
        if media_type == "video":
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return f"{w}x{h}"
        else:
            img = cv2.imread(file_path)
            if img is not None:
                h, w = img.shape[:2]
                return f"{w}x{h}"
    except Exception:
        pass
    return "unknown"


@app.post("/evaluate")
async def evaluate(
    file: UploadFile = File(...),
    media_type: str = Form(...),
    task_type: str = Form(...),
    model: str = Form(...),
    model_name: str = Form(""),
    backbone: str = Form(""),
    requested_models: str = Form(""),
):
    logger.info(f"[INFO] Evaluate: model={model}, file={file.filename}")

    if model not in model_cache:
        raise HTTPException(status_code=400, detail=f"Model not loaded: {model}")

    cached = model_cache[model]
    dl_model = cached["model"]
    dl_config = cached["config"]
    mos_min = cached["mos_min"]
    mos_max = cached["mos_max"]
    dataset = cached.get("dataset", "unknown")

    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    try:
        with open(tmp_fd, "wb") as f:
            shutil.copyfileobj(file.file, f)

        media_size = get_media_size(tmp_path, media_type)
        t_start = time.time()

        if model == "iqa":
            if media_type != "image":
                raise HTTPException(status_code=422, detail="iqa accepts image media only")
            result = predict_single(dl_model, tmp_path, dl_config, DEVICE, mos_min, mos_max)
        elif model == "vqa":
            if media_type != "video":
                raise HTTPException(status_code=422, detail="vqa accepts video media only")
            result = predict_single(dl_model, tmp_path, dl_config, DEVICE, mos_min, mos_max)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        elapsed_ms = (time.time() - t_start) * 1000

        raw_score = result.get("raw_score")
        dataset_mos = result.get("mos_score")
        response = {
            "file": file.filename,
            "media_type": media_type,
            "task_type": result.get("task_type", task_type),
            "model": model,
            "model_name": cached["model_name"],
            "backbone": cached["backbone"],
            "dataset": dataset,
            "mos": dataset_mos,
            "mos_score": dataset_mos,
            "raw_score": raw_score,
            "normalized_score": raw_score,
            "score": raw_score,
            "overall": dataset_mos,
            "dataset_mos": dataset_mos,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "inference_ms": round(elapsed_ms, 2),
            "latency_ms": round(elapsed_ms, 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "media_size": media_size,
            "metrics": {"plcc": None, "srocc": None, "rmse": None},
            "_debug_raw_score": raw_score,
            "_debug_dataset_mos": dataset_mos,
        }

        logger.info(
            f"[OK] Inference: raw={raw_score}, dataset_mos={dataset_mos}, "
            f"range=[{mos_min}, {mos_max}]"
        )
        return response

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception(f"[ERROR] Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "deploy.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
