#!/usr/bin/env python
# -*- coding: utf-8 -*-
# deploy/cli.py
"""
Inference CLI – Image/Video Quality Assessment

Automatically routes images to the IQA model and videos to the VQA model.

Usage:
    # Single file inference
    uv run python -m deploy.cli -i test.jpg
    uv run python -m deploy.cli -i test.mp4

    # Directory batch inference
    uv run python -m deploy.cli -i ./samples/

    # Custom checkpoint
    uv run python -m deploy.cli -i test.jpg -c ./best_model.pt

Examples:
    # IQA inference using the default model
    uv run python -m deploy.cli -i examples/photo.jpg

    # VQA inference
    uv run python -m deploy.cli -i examples/video.mp4

    # Custom MOS range (for denormalization)
    uv run python -m deploy.cli -i test.jpg --mos_min 0 --mos_max 5

Args:
    -i, --input        Input file or directory path
    -c, --checkpoint   Custom model checkpoint (automatically selected if omitted)
    --mos_min          Minimum MOS value for denormalization
    --mos_max          Maximum MOS value for denormalization
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from deploy.core.runtime_config import cfg, ensure_runtime_dirs
from deploy.core.model_loader import load_checkpoint, select_checkpoint
from deploy.core.inference import predict_batch


REPORTS_DIR = cfg.resolve(cfg.reports_dir)
SUPPORTED_EXTS = {ext.lower() for ext in (cfg.image_exts | cfg.video_exts)}


def collect_targets(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTS else []

    targets = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(targets, key=lambda path: str(path).lower())


def default_output_path(input_path: Path, task_group: str) -> Path:
    task_root = f"{task_group}-test" if task_group in {"iqa", "vqa"} else "test"
    report_dir = REPORTS_DIR / task_root
    report_dir.mkdir(parents=True, exist_ok=True)
    name = input_path.name if input_path.is_dir() else input_path.stem
    if not name:
        name = "batch"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return report_dir / f"{name}_{timestamp}.json"


def resolve_task_group(config: object) -> str:
    task_type = getattr(config, "task_type", None)
    if not task_type and hasattr(config, "dataset"):
        dataset_cfg = getattr(config, "dataset", None)
        task_type = getattr(dataset_cfg, "task_type", None)
    normalized = str(task_type or "").strip().lower()
    if normalized == "iqa":
        return "iqa"
    if normalized == "vqa":
        return "vqa"
    return "test"


def ensure_input_directory(input_path: Path) -> None:
    if not input_path.exists() and input_path.suffix == "":
        input_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="IQA/VQA Inference CLI")
    parser.add_argument("-i", "--input", type=str, required=True, help="File or directory")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--mos_min", type=float, default=None)
    parser.add_argument("--mos_max", type=float, default=None)

    args = parser.parse_args()
    device = cfg.default_device

    input_path = Path(args.input)
    ensure_runtime_dirs()
    ensure_input_directory(input_path)

    # Collect supported files with case-insensitive suffix matching
    targets = collect_targets(input_path)

    if not targets:
        logger.error("[ERROR] No files found")
        sys.exit(1)

    # Select and load model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else select_checkpoint(targets, cfg)
    if not ckpt_path.exists():
        logger.error(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, config = load_checkpoint(ckpt_path, device)

    # Run inference
    results = predict_batch(model, targets, config, device, args.mos_min, args.mos_max)

    # Output
    task_group = resolve_task_group(config)
    out_path = default_output_path(input_path, task_group)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"[OK] Saved to: {out_path}")

    # Print summary
    success = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    logger.info(f"[OK] Success: {len(success)}, Failed: {len(failed)}")


if __name__ == "__main__":
    main()
