# src/models/torchvision_cache.py
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import torch
import torch.nn as nn
from loguru import logger

from src.config.schemas import Config


_TORCHVISION_HASH_RE = re.compile(r"-([0-9a-f]{8,64})\.(?:pth|pt)$")


def _expected_hash_from_filename(path: Path) -> str | None:
    match = _TORCHVISION_HASH_RE.search(path.name)
    return match.group(1) if match else None


def _sha256_hexdigest(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None

    return digest.hexdigest()


def _sha256_prefix_matches(path: Path) -> bool:
    expected_hash = _expected_hash_from_filename(path)
    if expected_hash is None:
        return True

    actual_hash = _sha256_hexdigest(path)
    return actual_hash is not None and actual_hash.startswith(expected_hash)


def _prune_invalid_torch_checkpoints(checkpoints_dir: Path) -> None:
    if not checkpoints_dir.exists():
        return

    for checkpoint in checkpoints_dir.iterdir():
        if not checkpoint.is_file() or checkpoint.suffix not in {".pt", ".pth"}:
            continue
        if _sha256_prefix_matches(checkpoint):
            continue
        logger.warning(f"Removing invalid cached torchvision weight: {checkpoint}")
        try:
            checkpoint.unlink()
        except OSError:
            continue


def _checkpoint_name_from_weights(weights: object) -> str | None:
    url = getattr(weights, "url", None)
    if not url:
        return None
    return Path(urlparse(url).path).name


def _cached_weight_path(weights: object) -> Path | None:
    checkpoint_name = _checkpoint_name_from_weights(weights)
    if checkpoint_name is None:
        return None
    return Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name


def _remove_cached_weight_file(weights: object) -> Path | None:
    checkpoint_path = _cached_weight_path(weights)
    if checkpoint_path is None:
        return None
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        return checkpoint_path
    except OSError:
        return checkpoint_path
    return checkpoint_path


def _copy_existing_torch_checkpoints(source_dir: Path, target_dir: Path) -> None:
    """Reuse valid checkpoints already present in another torch cache."""
    if not source_dir.exists():
        return

    try:
        if source_dir.resolve() == target_dir.resolve():
            return
    except OSError:
        return

    for source in source_dir.iterdir():
        if not source.is_file() or source.suffix not in {".pt", ".pth"}:
            continue
        if not _sha256_prefix_matches(source):
            logger.warning(f"Skipping invalid cached torchvision weight: {source}")
            continue

        target = target_dir / source.name
        if target.exists():
            continue

        try:
            shutil.copy2(source, target)
        except OSError:
            continue


def configure_torch_weight_cache(cfg: Config) -> Path:
    """
    Keep torchvision pretrained weights in the configured project cache.

    Torchvision downloads weights through torch.hub. Binding torch.hub to
    cfg.paths.cache_dir keeps Docker and local runs on the same cache layout.
    """
    default_cache_root = Path.home() / ".cache" / "torch"
    env_cache_root = Path(os.environ["TORCH_HOME"]).expanduser() if os.environ.get("TORCH_HOME") else None
    cache_root = cfg.paths.resolve(cfg.paths.cache_dir) / "torch"
    os.environ["TORCH_HOME"] = str(cache_root)

    hub_dir = cache_root / "hub"
    checkpoints_dir = hub_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _prune_invalid_torch_checkpoints(checkpoints_dir)
    for source_root in (default_cache_root, env_cache_root):
        if source_root is None:
            continue
        _copy_existing_torch_checkpoints(source_root / "hub" / "checkpoints", checkpoints_dir)
    torch.hub.set_dir(str(hub_dir))
    logger.info(f"Torchvision pretrained weight cache: {checkpoints_dir}")
    return checkpoints_dir


def build_torchvision_model(builder: Callable[..., nn.Module], weights: object | None) -> nn.Module:
    try:
        return builder(weights=weights)
    except RuntimeError as exc:
        if weights is None or "invalid hash value" not in str(exc):
            raise

        checkpoint_path = _remove_cached_weight_file(weights)
        logger.warning(f"Invalid cached torchvision weight removed: {checkpoint_path}. Retrying download.")
        try:
            return builder(weights=weights)
        except RuntimeError as retry_exc:
            if "invalid hash value" in str(retry_exc):
                checkpoint_path = _remove_cached_weight_file(weights)
                raise RuntimeError(
                    "Torchvision pretrained weight failed hash validation after retry. "
                    f"Cache file removed: {checkpoint_path}. "
                    "The download source, mirror, or proxy is likely returning corrupt content."
                ) from retry_exc
            raise


__all__ = ["build_torchvision_model", "configure_torch_weight_cache"]
