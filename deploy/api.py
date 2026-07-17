#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI 多模型推理服务（统一0-5分尺度 + 媒体尺寸 + 跨媒体策略）
启动： uv run python -m deploy.api
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ---------- 路径处理 ----------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deploy.infer import (
    load_checkpoint,
    predict_single,
    predict_with_resnet_style,
    image_to_video_tensor,
    Preprocessor,
)

# ---------- 配置 ----------
DEFAULT_IQA_MODEL_PATH = Path(__file__).resolve().parent / "iqa-models" / "tid2013_best.pt"
DEFAULT_VQA_MODEL_PATH = Path(__file__).resolve().parent / "vqa-models" / "konvid_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 数据集特定的 MOS 反归一化参数（仅用于 dataset_mos）----------
DATASET_MOS_PARAMS = {
    "TID2013": {"mos_min": 0.242, "mos_max": 7.214},
    "KoNViD-1k": {"mos_min": 1.220, "mos_max": 4.640},
}

# ---------- FastAPI 实例 ----------
app = FastAPI(title="Deep-VQA Unified MOS API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 全局模型缓存 ----------
model_cache: Dict[str, Dict[str, Any]] = {}


def get_mos_params(config: Dict[str, Any], model_id: str) -> tuple:
    """获取模型对应的 MOS 反归一化参数（优先 config，回退到算法组提供的默认值）"""
    dataset_info = config.get("dataset_info", {}) or {}
    mos_min = dataset_info.get("mos_min")
    mos_max = dataset_info.get("mos_max")

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
    """服务启动时加载所有模型到内存"""
    logger.info(f"使用设备: {DEVICE}")

    if DEFAULT_IQA_MODEL_PATH.exists():
        logger.info(f"加载 IQA 模型: {DEFAULT_IQA_MODEL_PATH}")
        iqa_model, iqa_config = load_checkpoint(DEFAULT_IQA_MODEL_PATH, device=DEVICE)
        mos_min, mos_max = get_mos_params(iqa_config, "resnet_iqa")
        model_cache["resnet_iqa"] = {
            "model": iqa_model,
            "config": iqa_config,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "num_frames": iqa_config.get("model", {}).get("num_frames", 8),
            "input_size": iqa_config.get("model", {}).get("input_size", 224),
            "dataset": "TID2013",
        }
        logger.success(f"✅ IQA 模型加载完成 (TID2013 MOS 范围: {mos_min:.3f}~{mos_max:.3f})")
    else:
        logger.error(f"❌ IQA 模型未找到: {DEFAULT_IQA_MODEL_PATH}")

    if DEFAULT_VQA_MODEL_PATH.exists():
        logger.info(f"加载 VQA 模型: {DEFAULT_VQA_MODEL_PATH}")
        vqa_model, vqa_config = load_checkpoint(DEFAULT_VQA_MODEL_PATH, device=DEVICE)
        mos_min, mos_max = get_mos_params(vqa_config, "timeswin_vqa")
        model_cache["timeswin_vqa"] = {
            "model": vqa_model,
            "config": vqa_config,
            "mos_min": mos_min,
            "mos_max": mos_max,
            "num_frames": vqa_config.get("model", {}).get("num_frames", 8),
            "input_size": vqa_config.get("model", {}).get("input_size", 224),
            "dataset": "KoNViD-1k",
        }
        logger.success(f"✅ VQA 模型加载完成 (KoNViD-1k MOS 范围: {mos_min:.3f}~{mos_max:.3f})")
    else:
        logger.error(f"❌ VQA 模型未找到: {DEFAULT_VQA_MODEL_PATH}")

    if not model_cache:
        raise RuntimeError("没有成功加载任何模型，服务启动失败")


@app.on_event("startup")
async def startup_event():
    load_all_models()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": list(model_cache.keys()),
        "device": DEVICE,
    }


def get_media_size(file_path: str, media_type: str) -> str:
    """提取图片或视频的分辨率"""
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
    return "未知"


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
    print(f">>> 收到评价请求 | 模型: {model} | 文件: {file.filename}")
    logger.info(f"evaluate 函数被调用：model={model}, file={file.filename}")

    if model not in model_cache:
        raise HTTPException(status_code=400, detail=f"未加载的模型: {model}")

    cached = model_cache[model]
    dl_model = cached["model"]
    dl_config = cached["config"]
    mos_min = cached["mos_min"]
    mos_max = cached["mos_max"]
    num_frames = cached.get("num_frames", 8)
    input_size = cached.get("input_size", 224)
    dataset = cached.get("dataset", "未知")

    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with open(tmp_fd, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 提取媒体尺寸
        media_size = get_media_size(tmp_path, media_type)

        t_start = time.time()

        # ---------- 根据模型和媒体类型选择推理策略 ----------
        if model == "resnet_iqa":
            # IQA 模型：图片直推，视频抽帧平均
            result = predict_with_resnet_style(
                dl_model, tmp_path, dl_config, DEVICE, mos_min, mos_max
            )
        elif model == "timeswin_vqa":
            # VQA 模型：视频直推，图片扩展为伪视频
            if media_type == "image":
                preprocessor = Preprocessor(num_frames=num_frames, input_size=input_size)
                img_tensor = preprocessor._process_image(tmp_path)
                video_tensor = image_to_video_tensor(img_tensor, num_frames)
                data_tensor = video_tensor.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = dl_model(data_tensor).float()
                    if output.ndim > 1 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                    raw_score = float(output.flatten()[0].cpu().item())

                # 使用数据集参数反归一化（仅用于 dataset_mos）
                dataset_mos = round(raw_score * (mos_max - mos_min) + mos_min, 4)
                result = {
                    "file": str(tmp_path),
                    "raw_score": round(raw_score, 6),
                    "mos_score": dataset_mos,   # 临时存储数据集 MOS
                    "task_type": dl_config.get("task_type", "vqa"),
                    "model_name": dl_config.get("model", {}).get("name", "IQAVQANet"),
                }
            else:
                result = predict_single(
                    dl_model, tmp_path, dl_config, DEVICE, mos_min, mos_max
                )
        else:
            raise HTTPException(status_code=400, detail=f"未知模型: {model}")

        elapsed_ms = (time.time() - t_start) * 1000

        raw_score = result.get("raw_score")
        dataset_mos = result.get("mos_score")   # 来自推理函数的数据集 MOS

        # ---------- 计算统一 0-5 分（raw_score * 5）----------
        unified_mos = round(raw_score * 5.0, 4) if raw_score is not None else None

        # 如果因为某种原因 raw_score 缺失，用数据集的 mos 兜底（但通常不会）
        if unified_mos is None and dataset_mos is not None:
            # 极其罕见的情况，用数据集 MOS 除以其最大值近似映射到 0-5
            unified_mos = round(dataset_mos / (mos_max if mos_max else 5.0) * 5.0, 4)

        response = {
            "file": file.filename,
            "media_type": media_type,
            "task_type": result.get("task_type", task_type),
            "model": model,
            "model_name": result.get("model_name", model_name),
            "backbone": backbone,
            "dataset": dataset,
            "mos": unified_mos,                     # ← 前端直接显示的统一0-5分
            "mos_score": unified_mos,
            "raw_score": raw_score,
            "score": raw_score,
            "overall": unified_mos,
            "dataset_mos": dataset_mos,             # ← 原始数据集尺度 MOS（仅供调试）
            "inference_ms": round(elapsed_ms, 2),
            "latency_ms": round(elapsed_ms, 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "media_size": media_size,
            "metrics": {
                "plcc": None,
                "srocc": None,
                "rmse": None,
            },
            "_debug_backend_loaded_models": list(model_cache.keys()),
            "_debug_actual_model_used": model,
            "_debug_raw_score": raw_score,
            "_debug_unified_mos": unified_mos,
            "_debug_dataset_mos": dataset_mos,
        }

        print(f">>> 推理完成 | raw={raw_score} | 统一MOS={unified_mos} | 数据集MOS={dataset_mos} | 尺寸={media_size}")
        return response

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception(f"推理时发生异常: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")
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