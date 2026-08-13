# src/core/trainer.py
import gc
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.config.schemas import Config
from src.core.engine import TrainerEngine
from src.data.eda.split import (
    infer_group_labels,
    make_group_kfold_splits,
    validate_group_split_isolation,
)
from src.data.preprocessing import (
    rgb_array_to_imagenet_tensor,
    rgb_video_array_to_imagenet_tensor,
)

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("Decord not available, using OpenCV for video loading.")

from src.models.iqavqa_model import IQAVQANet
from src.models.losses import IQAVQALoss
from src.utils.checkpointing import select_best_checkpoint


def release_torch_cache(func: Callable[..., Any]) -> Callable[..., Any]:
    """Release CUDA/cache resources after a training or evaluation stage."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return wrapper


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
        self.input_size = cfg.model.input_size
        self.data_type = cfg.dataset.data_type

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
            raise FileNotFoundError(f"Failed to resolve {sample_id}") from e

        try:
            if is_video:
                data_tensor = self._read_video(str(target_path))
            else:
                data_tensor = self._read_image(str(target_path))
        except Exception as e:
            raise RuntimeError(f"Failed to read {target_path}") from e

        label = torch.tensor(row.get("normalized_score", row["mos"]), dtype=torch.float32)
        payload = {"data": data_tensor, "label": label}

        if self.traditional_cols:
            payload["traditional"] = {
                col: torch.tensor(row[col], dtype=torch.float32)
                for col in self.traditional_cols
            }

        return payload

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

                tensor = rgb_video_array_to_imagenet_tensor(frames, input_size=self.input_size)

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
            raise ValueError(f"Failed to open video: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = range(self.num_frames)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No readable frames in video: {path}")

        video_np = np.stack(frames)
        tensor = rgb_video_array_to_imagenet_tensor(video_np, input_size=self.input_size)

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

        return rgb_array_to_imagenet_tensor(img, input_size=self.input_size)


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


    def execute_cross_validation(self, evaluator: Any) -> int:
        """Execute K-fold cross-validation."""
        n_splits = self.cfg.preprocessing.k_fold
        dataset_key = self.cfg.dataset.registry_key or self.cfg.dataset.name
        logger.info(f"Starting {n_splits}-fold cross-validation...")

        clean_df = self._prepare_data()

        test_df = None
        if "split" in clean_df.columns:
            active_df = clean_df[clean_df["split"].isin(["train", "val"])].reset_index(drop=True)
            test_df = clean_df[clean_df["split"] == "test"]
            logger.info(f"Test set isolated: {len(test_df)} samples")
            if not validate_group_split_isolation(
                clean_df,
                dataset_name=dataset_key,
                context="train/val/test",
            ):
                raise RuntimeError("Group leakage detected between train/val/test splits")
        else:
            active_df = clean_df
            logger.warning("No 'split' column found, data leakage risk.")

        original_base_filename = getattr(evaluator, "base_filename", "model")
        splits, split_strategy = make_group_kfold_splits(
            df=active_df,
            n_splits=n_splits,
            random_state=42,
            dataset_name=dataset_key,
        )
        if not splits:
            raise RuntimeError("No cross-validation splits were generated")

        logger.info(f"Using {split_strategy} for cross-validation")
        completed_folds = 0

        for fold, (train_idx, val_idx) in enumerate(splits):
            fold_num = fold + 1
            completed_folds = fold_num
            logger.info(f"Fold {fold_num}/{len(splits)}")

            evaluator.base_filename = f"{original_base_filename}_fold{fold_num}"

            train_sub = active_df.iloc[train_idx].reset_index(drop=True)
            val_sub = active_df.iloc[val_idx].reset_index(drop=True)
            fold_df = pd.concat(
                [
                    train_sub.assign(_fold_split="train"),
                    val_sub.assign(_fold_split="val"),
                ],
                ignore_index=True,
            )
            if not validate_group_split_isolation(
                fold_df,
                split_col="_fold_split",
                dataset_name=dataset_key,
                context=f"fold {fold_num}",
            ):
                raise RuntimeError(f"Group leakage detected in fold {fold_num}")

            train_groups = infer_group_labels(train_sub, dataset_name=dataset_key).nunique()
            val_groups = infer_group_labels(val_sub, dataset_name=dataset_key).nunique()
            logger.info(
                f"Fold {fold_num}: Train={len(train_sub)} samples/{train_groups} groups, "
                f"Val={len(val_sub)} samples/{val_groups} groups"
            )

            self._run_fold(
                train_sub=train_sub,
                val_sub=val_sub,
                evaluator=evaluator,
                current_fold=fold_num,
            )

            logger.info(f"Fold {fold_num} completed")

            if self.fast_run:
                logger.info("Fast run mode: stopping after first fold.")
                break

        if test_df is not None and not test_df.empty:
            logger.info("Evaluating on test set...")
            self._evaluate_test_set(test_df, evaluator, original_base_filename)

        return completed_folds


    @release_torch_cache
    def _run_fold(
        self,
        train_sub: pd.DataFrame,
        val_sub: pd.DataFrame,
        evaluator: Any,
        current_fold: int,
    ) -> None:
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=optimizer_cfg.epochs,
            eta_min=1e-6,
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
            current_fold=current_fold,
        )


    def _prepare_data(self) -> pd.DataFrame:
        """Filter outliers and prepare data for cross-validation."""
        clean_df = self.eda_df.copy()

        if "is_outlier" in clean_df.columns:
            clean_df = clean_df[~clean_df["is_outlier"]].reset_index(drop=True)
            logger.info(f"Filtered outliers, remaining {len(clean_df)} samples")

        return clean_df



    @release_torch_cache
    def _evaluate_test_set(
        self,
        test_df: pd.DataFrame,
        evaluator: Any,
        base_filename: str,
    ) -> None:
        """Evaluate on the held-out test set."""
        model = IQAVQANet(cfg=self.cfg, load_pretrained_backbone=False).to(self.device)
        dataset_key = self.cfg.dataset.registry_key or self.cfg.dataset.name
        save_dir = self.cfg.paths.model_outputs_dir(dataset_key)
        best_model_path = select_best_checkpoint(
            model_outputs_dir=save_dir,
            base_fn=base_filename,
            monitor=self.cfg.train.checkpoint.monitor.lower(),
            mode=self.cfg.train.checkpoint.mode.lower(),
        )
        test_base_filename = f"{base_filename}_test"

        logger.info(f"Loading best model for test set: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])

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

        evaluator.base_filename = test_base_filename
        evaluator.evaluate(y_true_real, y_pred_real, save_manifest=True)
