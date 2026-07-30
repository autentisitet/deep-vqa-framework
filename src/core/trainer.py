# src/core/trainer.py
import copy
import gc
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("Decord not available, using OpenCV for video loading.")

from src.config.schemas import Config
from src.core.engine import TrainerEngine
from src.models.iqavqa_net import IQAVQALoss, IQAVQANet


def worker_init_fn(worker_id: int) -> None:
    """Set random seed for each DataLoader worker."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class ImageVideoDataset(Dataset):
    """
    Dataset that supports both images and videos.
    Reads DataFrames from DataEDA and decodes images/video frames on the fly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        cfg: Config,
        resolver: Any,
        transform: Any = None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.resolver = resolver
        self.transform = transform

        self.num_frames = cfg.model.num_frames

        self.traditional_cols = [
            c for c in ["ssim", "vif", "dlm", "vmaf", "niqe"]
            if c in self.df.columns
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        sample_id = str(row["sample_id"]).strip()

        try:
            asset = self.resolver.resolve(sample_id)
            target_path = asset.path
            is_video = asset.is_video
        except Exception as e:
            logger.error(f"Failed to resolve {sample_id}: {e}")
            return self._empty_payload()

        try:
            if is_video:
                data_tensor = self._read_video(str(target_path))
            else:
                data_tensor = self._read_image(str(target_path))
        except Exception as e:
            logger.error(f"Failed to read {target_path}: {e}")
            return self._empty_payload()

        label = torch.tensor(row.get("normalized_score", row["mos"]), dtype=torch.float32)
        payload = {"data": data_tensor, "label": label}

        if self.traditional_cols:
            payload["traditional"] = {
                col: torch.tensor(row[col], dtype=torch.float32)
                for col in self.traditional_cols
            }

        return payload

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "data": torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32),
            "label": torch.tensor(0.0, dtype=torch.float32),
        }

    def _read_video(self, path: str) -> torch.Tensor:
        """Read video using Decord with OpenCV fallback."""
        if DECORD_AVAILABLE:
            try:
                vr = VideoReader(path, ctx=cpu(0))
                total_frames = len(vr)

                if total_frames >= self.num_frames:
                    indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
                else:
                    indices = list(range(total_frames))

                frames = vr.get_batch(indices).asnumpy()
                if frames.size == 0:
                    raise ValueError("Empty frames")

                resized = [cv2.resize(f, (224, 224)) for f in frames]
                video_np = np.stack(resized)
                tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

                if tensor.size(0) < self.num_frames:
                    pad = tensor[-1].unsqueeze(0).repeat(self.num_frames - tensor.size(0), 1, 1, 1)
                    tensor = torch.cat([tensor, pad], dim=0)

                return tensor
            except Exception as e:
                logger.debug(f"Decord failed for {path}: {e}")

        return self._read_video_opencv(path)

    def _read_video_opencv(self, path: str) -> torch.Tensor:
        """OpenCV fallback for video reading."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        video_np = np.stack(frames)
        tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

        if tensor.size(0) < self.num_frames:
            pad = tensor[-1].unsqueeze(0).repeat(self.num_frames - tensor.size(0), 1, 1, 1)
            tensor = torch.cat([tensor, pad], dim=0)

        return tensor

    def _read_image(self, path: str) -> torch.Tensor:
        """Read and preprocess a single image."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            return self.transform(img)

        img = cv2.resize(img, (224, 224))
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class TrainerExecutionPipeline:
    """
    Manages K-fold cross-validation training pipeline.
    """

    def __init__(
        self,
        cfg: Config,
        eda_df: pd.DataFrame,
        data_dir: Path,
        resolver: Any,
        fast_run: bool = False,
    ):
        self.cfg = cfg
        self.eda_df = eda_df
        self.data_dir = Path(data_dir).resolve()
        self.resolver = resolver
        self.fast_run = fast_run

        self.task_type = cfg.task_type
        self.is_video = cfg.dataset.data_type == "video"

        self.batch_size = cfg.preprocessing.batch_size
        self.num_workers = cfg.preprocessing.num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def execute_cross_validation(self, evaluator: Any) -> None:
        """Execute K-fold cross-validation."""
        n_splits = self.cfg.preprocessing.k_fold
        logger.info(f"Starting {n_splits}-fold cross-validation...")

        clean_df = self._prepare_data()

        test_df = None
        if "split" in clean_df.columns:
            active_df = clean_df[clean_df["split"].isin(["train", "val"])].reset_index(drop=True)
            test_df = clean_df[clean_df["split"] == "test"]
            logger.info(f"Test set isolated: {len(test_df)} samples")
        else:
            active_df = clean_df
            logger.warning("No 'split' column found, data leakage risk.")

        original_base_filename = getattr(evaluator, "base_filename", "model")
        original_cfg = self.cfg

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(active_df)):
            fold_num = fold + 1
            logger.info(f"Fold {fold_num}/{n_splits}")

            evaluator.base_filename = original_base_filename

            train_sub = active_df.iloc[train_idx]
            val_sub = active_df.iloc[val_idx]

            train_dataset = ImageVideoDataset(
                train_sub, self.data_dir, cfg=self.cfg, resolver=self.resolver
            )
            val_dataset = ImageVideoDataset(
                val_sub, self.data_dir, cfg=self.cfg, resolver=self.resolver
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.cfg.system.pin_memory,
                worker_init_fn=worker_init_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.cfg.system.pin_memory,
            )

            model = IQAVQANet(cfg=self.cfg).to(self.device)

            optimizer_cfg = self.cfg.train
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
            )

            epochs = optimizer_cfg.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            criterion = IQAVQALoss(cfg=self.cfg)

            engine = TrainerEngine(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=evaluator,
                cfg=self.cfg,
                scheduler=scheduler,
                device=self.device,
            )

            engine.fit(
                train_loader,
                val_loader,
                current_fold=fold_num,
            )

            logger.info(f"Fold {fold_num} completed")

            del model, optimizer, scheduler, criterion, engine
            torch.cuda.empty_cache()
            gc.collect()

            if self.fast_run:
                logger.info("Fast run mode: stopping after first fold.")
                break

        if test_df is not None and not test_df.empty:
            logger.info("Evaluating on test set...")
            self._evaluate_test_set(test_df, evaluator, original_base_filename, n_splits)



    def _prepare_data(self) -> pd.DataFrame:
        """Filter outliers and prepare data for cross-validation."""
        clean_df = self.eda_df.copy()

        if "is_outlier" in clean_df.columns:
            clean_df = clean_df[~clean_df["is_outlier"]].reset_index(drop=True)
            logger.info(f"Filtered outliers, remaining {len(clean_df)} samples")

        return clean_df



    def _evaluate_test_set(
        self,
        test_df: pd.DataFrame,
        evaluator: Any,
        base_filename: str,
        n_splits: int,
    ) -> None:
        """Evaluate on the held-out test set."""
        model = IQAVQANet(cfg=self.cfg).to(self.device)
        save_dir = self.cfg.paths.model_outputs_dir(self.cfg.dataset.name)
        best_model_path = save_dir / f"{base_filename}_fold{n_splits}_best.pt"

        if best_model_path.exists():
            logger.info(f"Loading best model: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.warning("No best model found, using random weights.")

        model.eval()

        test_dataset = ImageVideoDataset(
            test_df, self.data_dir, cfg=self.cfg, resolver=self.resolver
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.cfg.system.pin_memory,
        )

        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for batch in test_loader:
                data = batch["data"].to(self.device)
                target = batch["label"].to(self.device)
                output = model(data)
                y_true_list.append(target.cpu())
                y_pred_list.append(output.cpu())

        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()

        # Denormalize if MOS range is available
        mos_min = self.cfg.dataset.mos_min
        mos_max = self.cfg.dataset.mos_max

        if mos_min is not None and mos_max is not None:
            y_true_real = y_true * (mos_max - mos_min) + mos_min
            y_pred_real = y_pred * (mos_max - mos_min) + mos_min
            logger.info(f"Denormalized to original MOS range [{mos_min:.3f}, {mos_max:.3f}]")
        else:
            y_true_real, y_pred_real = y_true, y_pred
            logger.warning("No MOS range found, RMSE is in [0,1] scale.")

        evaluator.evaluate(y_true_real, y_pred_real, save_manifest=True)

        del model
        torch.cuda.empty_cache()
        gc.collect()