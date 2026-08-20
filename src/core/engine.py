# src/core/engine.py
from typing import Any, Optional

import re
import shutil
from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.schemas import Config

try:
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast

    def make_grad_scaler(device_type: str, enabled: bool) -> AmpGradScaler:
        return AmpGradScaler(device_type, enabled=enabled)

    def autocast_context(device_type: str, enabled: bool):
        return amp_autocast(device_type=device_type, enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as amp_autocast

    def make_grad_scaler(device_type: str, enabled: bool) -> AmpGradScaler:
        return AmpGradScaler(enabled=enabled)

    def autocast_context(device_type: str, enabled: bool):
        return amp_autocast(enabled=enabled)


def progress_bar(iterable: Any, *, desc: str) -> tqdm:
    total = len(iterable) if hasattr(iterable, "__len__") else None
    miniters = max(1, math.ceil(total * 0.02)) if total else None
    return tqdm(iterable, desc=desc, miniters=miniters)


class TrainerEngine:
    """
    General Training/Validation Engine.

    Responsibilities:
    - Training loop (forward/backward, gradient update)
    - Mixed precision training (AMP)
    - Validation and metric collection
    - Checkpoint saving and early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        evaluator: Any,
        cfg: Config,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluator = evaluator
        self.cfg = cfg
        self.scheduler = scheduler

        train_cfg = cfg.train
        self.epochs = train_cfg.epochs
        self.grad_clip = train_cfg.grad_clip
        self.gradient_accumulation_steps = train_cfg.gradient_accumulation_steps

        self.use_amp = cfg.system.amp
        self.scaler = make_grad_scaler(self.device_type, enabled=self.use_amp and self.device_type == "cuda")

        # Early stopping
        early_stop_cfg = train_cfg.early_stop
        self.early_stop_enabled = early_stop_cfg.enabled
        self.patience = early_stop_cfg.patience
        self.early_stop_monitor = early_stop_cfg.monitor.lower()
        self.early_stop_mode = early_stop_cfg.mode

        self.early_stop_counter = 0
        self.best_early_stop_score = float("-inf") if self.early_stop_mode == "max" else float("inf")

        # Checkpoint
        checkpoint_cfg = train_cfg.checkpoint
        self.checkpoint_monitor = checkpoint_cfg.monitor.lower()
        self.checkpoint_secondary_monitor = checkpoint_cfg.secondary_monitor.lower()
        self.checkpoint_mode = checkpoint_cfg.mode
        self.best_checkpoint_score = float("-inf") if self.checkpoint_mode == "max" else float("inf")
        self.best_checkpoint_secondary_score = float("-inf")

        logger.info(
            f"TrainerEngine initialized. Device: {self.device} | "
            f"Monitor: {self.checkpoint_monitor} | AMP: {self.use_amp}"
        )


    def _sanitize_for_serialization(self, obj):
        """Recursively convert Path objects to strings."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_serialization(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_serialization(v) for v in obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "model_dump"):
            return self._sanitize_for_serialization(obj.model_dump())
        else:
            return obj



    def resume_training(self, checkpoint_path: str) -> int:
        """Resume training from checkpoint."""
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")
            return 1

        logger.info(f"Loading checkpoint: {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location=self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        logger.info("  - Model weights loaded")

        if "optimizer" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("  - Optimizer state loaded")

        if "scheduler" in checkpoint and self.scheduler and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("  - Scheduler state loaded")

        resume_epoch = checkpoint.get("epoch", 0) + 1

        if "metrics" in checkpoint:
            hist_metrics = {k.lower(): v for k, v in checkpoint["metrics"].items()}
            self.best_checkpoint_score = hist_metrics.get(self.checkpoint_monitor, self.best_checkpoint_score)
            self.best_checkpoint_secondary_score = hist_metrics.get(
                self.checkpoint_secondary_monitor,
                self.best_checkpoint_secondary_score,
            )
            if self.early_stop_enabled:
                self.best_early_stop_score = hist_metrics.get(self.early_stop_monitor, self.best_early_stop_score)

        logger.info(f"Resuming from epoch {resume_epoch}")
        return resume_epoch



    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        accum_steps = self.gradient_accumulation_steps

        pbar = progress_bar(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for batch_idx, batch_data in enumerate(pbar):
            if isinstance(batch_data, dict):
                inputs = batch_data["data"].to(self.device)
                labels = batch_data["label"].to(self.device)
            else:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

            with autocast_context(self.device_type, enabled=self.use_amp and self.device_type == "cuda"):
                outputs = self.model(inputs)

            with autocast_context(self.device_type, enabled=False):
                outputs_f32 = outputs.float()
                if outputs_f32.ndim > 1 and outputs_f32.size(-1) == 1:
                    outputs_f32 = outputs_f32.squeeze(-1)
                labels = labels.view_as(outputs_f32).float()
                loss_dict = self.criterion(outputs_f32, labels)
                total_loss = loss_dict["total_loss"] / accum_steps

            self.scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += total_loss.item() * accum_steps

            log_interval = self.cfg.logging.log_interval
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    "Loss": f"{total_loss.item() * accum_steps:.4f}",
                    "Huber": f"{float(loss_dict.get('smooth_l1_loss', 0.0)):.4f}",
                    "Rank": f"{float(loss_dict.get('rank_loss', 0.0)):.4f}",
                })

        return running_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        epoch: int,
        train_loss: Optional[float] = None,
        save_manifest: bool = True,
    ) -> dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_trues = []
        traditional_metrics_payload = {}

        for batch_data in progress_bar(val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]"):
            if isinstance(batch_data, dict):
                inputs = batch_data["data"].to(self.device)
                labels = batch_data["label"].to(self.device)
                if "traditional" in batch_data:
                    for k, v in batch_data["traditional"].items():
                        traditional_metrics_payload.setdefault(k, []).extend(v.numpy())
            else:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

            with autocast_context(self.device_type, enabled=self.use_amp and self.device_type == "cuda"):
                outputs = self.model(inputs)

            with autocast_context(self.device_type, enabled=False):
                outputs_f32 = outputs.float()
                if outputs_f32.ndim > 1 and outputs_f32.size(-1) == 1:
                    outputs_f32 = outputs_f32.squeeze(-1)
                labels = labels.view_as(outputs_f32).float()
                loss_dict = self.criterion(outputs_f32, labels)

            running_loss += loss_dict["total_loss"].item()
            all_preds.extend(outputs_f32.cpu().numpy().flatten())
            all_trues.extend(labels.cpu().numpy().flatten())

        val_loss = running_loss / len(val_loader)
        traditional_metrics = (
            {k: np.array(v) for k, v in traditional_metrics_payload.items()} if traditional_metrics_payload else None
        )

        eval_fn = getattr(self.evaluator, "evaluate", None)
        if eval_fn is None:
            for alt_name in ["execute", "compute", "run"]:
                if hasattr(self.evaluator, alt_name):
                    eval_fn = getattr(self.evaluator, alt_name)
                    break

        if eval_fn is None:
            raise AttributeError("Evaluator missing evaluate/execute method.")

        metrics = eval_fn(
            y_true=np.array(all_trues),
            y_pred=np.array(all_preds),
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            traditional_metrics=traditional_metrics,
            save_manifest=save_manifest,
            manifest_thresholds=self.cfg.train.manifest.thresholds,
            manifest_enabled=self.cfg.train.manifest.enabled,
        )

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 1,
        current_fold: int = 1,
    ) -> None:
        """Main training loop."""
        logger.info(f"Training started. Start epoch: {start_epoch}, Total epochs: {self.epochs}")

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            raw_metrics = self.evaluate(
                val_loader,
                epoch,
                train_loss=train_loss,
                save_manifest=True,
            )

            metrics = {k.lower(): v for k, v in raw_metrics.items()}
            val_loss = raw_metrics.get("val_loss", metrics.get("val_loss", 0.0))
            if val_loss == 0.0 and "loss" in metrics:
                val_loss = metrics["loss"]

            metrics["val_loss"] = val_loss
            metrics["loss"] = val_loss

            for indicator in ["srocc", "plcc", "rmse"]:
                if indicator in metrics:
                    metrics[f"val_{indicator}"] = metrics[indicator]

            val_srocc = metrics.get("val_srocc", 0.0)
            val_plcc = metrics.get("val_plcc", 0.0)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch} | LR: {current_lr:.6f} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"SROCC: {val_srocc:.4f} | PLCC: {val_plcc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | SROCC: {val_srocc:.4f} | PLCC: {val_plcc:.4f}"
                )

            should_stop = False

            # Early stopping uses the primary validation metric only.
            if self.early_stop_enabled:
                score = metrics.get(self.early_stop_monitor, val_loss)
                if self.early_stop_monitor not in metrics:
                    logger.warning(f"Monitor '{self.early_stop_monitor}' not found, using val_loss.")

                if self._is_improved(score, self.best_early_stop_score, self.early_stop_mode):
                    self.best_early_stop_score = score
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                if self.early_stop_counter >= self.patience:
                    logger.warning(
                        f"Early stopping triggered. No improvement in '{self.early_stop_monitor}' "
                        f"for {self.early_stop_counter} epochs."
                    )
                    should_stop = True

            # Checkpoint
            ckpt_score = metrics.get(self.checkpoint_monitor, 0.0)
            ckpt_secondary_score = metrics.get(self.checkpoint_secondary_monitor, 0.0)
            if self._is_primary_or_tiebreak_improved(
                ckpt_score,
                ckpt_secondary_score,
                self.best_checkpoint_score,
                self.best_checkpoint_secondary_score,
                self.checkpoint_mode,
            ):
                self.best_checkpoint_score = ckpt_score
                self.best_checkpoint_secondary_score = ckpt_secondary_score
                self._save_checkpoint(
                    epoch,
                    val_loss,
                    metrics,
                    is_best=True,
                    current_fold=current_fold,
                )

            if epoch == self.epochs:
                self._save_checkpoint(
                    epoch,
                    val_loss,
                    metrics,
                    is_best=False,
                    current_fold=current_fold,
                )

            if should_stop:
                break

            torch.cuda.empty_cache()



    def _is_improved(self, current: float, best: float, mode: str) -> bool:
        return current < best if mode == "min" else current > best

    def _is_primary_or_tiebreak_improved(
        self,
        current: float,
        current_secondary: float,
        best: float,
        best_secondary: float,
        mode: str,
    ) -> bool:
        if self._is_improved(current, best, mode):
            return True
        return current == best and current_secondary > best_secondary


    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        metrics: dict[str, Any],
        is_best: bool,
        current_fold: int = 1,
    ) -> None:
        """Save model checkpoint."""

        dataset_key = self.cfg.dataset.registry_key or self.cfg.dataset.name
        save_dir = self.cfg.paths.model_outputs_dir(dataset_key)
        save_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = self.cfg.train
        top_k = train_cfg.checkpoint.save_top_k

        base_name = self.evaluator.base_filename
        current_score = metrics.get(self.checkpoint_monitor, 0.0)
        current_secondary_score = metrics.get(self.checkpoint_secondary_monitor, 0.0)

        def extract_score(path: Path, monitor: str, fallback: float) -> float:
            match = re.search(rf"{re.escape(monitor)}(-?\d+\.\d+)", path.name.lower())
            if match:
                return float(match.group(1))
            return fallback

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            # Convert Path objects to strings for safe loading with weights_only=True
            "config": self._sanitize_for_serialization(self.cfg.model_dump()),
        }

        if is_best:
            if f"_fold{current_fold}" in base_name:
                clean_base = base_name
            else:
                clean_base = f"{base_name}_fold{current_fold}"

            pt_name = (
                f"{clean_base}_best_epoch{epoch}_{self.checkpoint_monitor}{current_score:.6f}_"
                f"{self.checkpoint_secondary_monitor}{current_secondary_score:.6f}.pt"
            )
            target_path = save_dir / pt_name
            torch.save(state, target_path)
            logger.info(f"Best checkpoint saved: {target_path.name}")

            all_best_pts = list(save_dir.glob(f"{clean_base}_best_epoch*.pt"))
            if len(all_best_pts) > top_k:
                reverse_flag = self.checkpoint_mode == "max"
                primary_fallback = float("-inf") if self.checkpoint_mode == "max" else float("inf")

                def checkpoint_sort_key(path: Path) -> tuple[float, float]:
                    primary = extract_score(path, self.checkpoint_monitor, primary_fallback)
                    secondary = extract_score(path, self.checkpoint_secondary_monitor, float("-inf"))
                    return (primary, secondary) if reverse_flag else (-primary, secondary)

                all_best_pts.sort(key=checkpoint_sort_key, reverse=True)
                for old_pt in all_best_pts[top_k:]:
                    old_pt.unlink()
                    logger.debug(f"Removed old checkpoint: {old_pt.name}")

            standard_best_path = save_dir / f"{clean_base}_best.pt"
            shutil.copy(target_path, standard_best_path)

        else:
            pt_name = f"{base_name}_epoch{epoch}_{self.checkpoint_monitor}{current_score:.4f}.pt"
            target_path = save_dir / pt_name
            torch.save(state, target_path)
            logger.info(f"Checkpoint saved: {target_path.name}")
