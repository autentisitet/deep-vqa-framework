#!/usr/bin/env python
# -*- coding: utf-8 -*-
# deploy/api.py
"""
FastAPI inference service.

Usage:
    uv run python -m deploy.api
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from deploy.core.runtime_config import cfg, ensure_runtime_dirs
from deploy.core.preprocessor import Preprocessor
from deploy.core.model_loader import load_checkpoint
from deploy.core.inference import (
    predict_single,
    predict_with_resnet_style,
    image_to_video_tensor,
)
from src.config.schemas import Config


# ---------- Configuration ----------
DEFAULT_IQA_MODEL_PATH = cfg.iqa_model_path
DEFAULT_VQA_MODEL_PATH = cfg.vqa_model_path
DEVICE = cfg.default_device



# ---------- MOS Parameters ----------
DATASET_MOS_PARAMS = {
    "TID2013": {"mos_min": 0.242, "mos_max": 7.214},
    "KoNViD-1k": {"mos_min": 1.220, "mos_max": 4.640},
}



# ---------- FastAPI ----------
app = FastAPI(title="Deep-VQA Unified MOS API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_cache: Dict[str, Dict[str, Any]] = {}


def get_mos_params(config: Config, model_id: str) -> tuple:
    """获取 MOS 反归一化参数"""
    dataset_cfg = getattr(config, "dataset", {}) or {}
    if hasattr(dataset_cfg, "model_dump"):
        dataset_cfg = dataset_cfg.model_dump()
    mos_min = dataset_cfg.get("mos_min")
    mos_max = dataset_cfg.get("mos_max")

    if mos_min is None or mos_max is None:
        if model_id == "resnet_iqa":
            params = DATASET_MOS_PARAMS["TID2013"]
        elif model_id == "timeswin_vqa":
            params = DATASET_MOS_PARAMS["KoNViD-1k"]
        else:
            params = {"mos_min": 0.0, "mos_max": 5.0}
        mos_min = params["mos_min"]
        mos_max = params["mos_max"]

    return mos_min, mos_max


def load_all_models():
    logger.info(f"[INFO] Using device: {DEVICE}")

    if DEFAULT_IQA_MODEL_PATH.exists():
        logger.info(f"[INFO] Loading IQA model: {DEFAULT_IQA_MODEL_PATH}")
        model, config_obj = load_checkpoint(DEFAULT_IQA_MODEL_PATH, device=DEVICE)
        mos_min, mos_max = get_mos_params(config_obj, "resnet_iqa")
        dataset_cfg = getattr(config_obj, "dataset", {}) or {}
        if hasattr(dataset_cfg, "model_dump"):
            dataset_cfg = dataset_cfg.model_dump()
        model_cache["resnet_iqa"] = {
            "model": model,
            "config": config_obj,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "dataset": dataset_cfg.get("name", "TID2013"),
        }
        logger.info(f"[OK] IQA loaded (MOS: {mos_min:.3f}~{mos_max:.3f})")
    else:
        logger.warning(f"[WARN] IQA model not found: {DEFAULT_IQA_MODEL_PATH}")

    if DEFAULT_VQA_MODEL_PATH.exists():
        logger.info(f"[INFO] Loading VQA model: {DEFAULT_VQA_MODEL_PATH}")
        model, config_obj = load_checkpoint(DEFAULT_VQA_MODEL_PATH, device=DEVICE)
        mos_min, mos_max = get_mos_params(config_obj, "timeswin_vqa")
        dataset_cfg = getattr(config_obj, "dataset", {}) or {}
        if hasattr(dataset_cfg, "model_dump"):
            dataset_cfg = dataset_cfg.model_dump()
        model_cache["timeswin_vqa"] = {
            "model": model,
            "config": config_obj,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "dataset": dataset_cfg.get("name", "KoNViD-1k"),
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

        if model == "resnet_iqa":
            result = predict_with_resnet_style(
                dl_model, tmp_path, dl_config, DEVICE, mos_min, mos_max
            )
        elif model == "timeswin_vqa":
            if media_type == "image":
                preprocessor = Preprocessor()
                img_tensor = preprocessor.process_image(tmp_path)
                video_tensor = image_to_video_tensor(img_tensor, cached.get("num_frames", 8))
                data_tensor = video_tensor.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = dl_model(data_tensor).float()
                    if output.ndim > 1 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                    raw_score = float(output.flatten()[0].cpu().item())

                dataset_mos = round(raw_score * (mos_max - mos_min) + mos_min, 4)
                result = {
                    "file": str(tmp_path),
                    "raw_score": round(raw_score, 6),
                    "mos_score": dataset_mos,
                    "task_type": dl_config.get("task_type", "vqa"),
                    "model_name": dl_config.get("model", {}).get("name", "IQAVQANet"),
                }
            else:
                result = predict_single(dl_model, tmp_path, dl_config, DEVICE, mos_min, mos_max)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        elapsed_ms = (time.time() - t_start) * 1000

        raw_score = result.get("raw_score")
        dataset_mos = result.get("mos_score")
        unified_mos = round(raw_score * 5.0, 4) if raw_score is not None else None

        if unified_mos is None and dataset_mos is not None:
            unified_mos = round(dataset_mos / (mos_max if mos_max else 5.0) * 5.0, 4)

        response = {
            "file": file.filename,
            "media_type": media_type,
            "task_type": result.get("task_type", task_type),
            "model": model,
            "model_name": result.get("model_name", model_name),
            "backbone": backbone,
            "dataset": dataset,
            "mos": unified_mos,
            "mos_score": unified_mos,
            "raw_score": raw_score,
            "score": raw_score,
            "overall": unified_mos,
            "dataset_mos": dataset_mos,
            "inference_ms": round(elapsed_ms, 2),
            "latency_ms": round(elapsed_ms, 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "media_size": media_size,
            "metrics": {"plcc": None, "srocc": None, "rmse": None},
            "_debug_raw_score": raw_score,
            "_debug_unified_mos": unified_mos,
            "_debug_dataset_mos": dataset_mos,
        }

        logger.info(f"[OK] Inference: raw={raw_score}, unified={unified_mos}")
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
