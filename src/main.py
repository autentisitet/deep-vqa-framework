import argparse
import os
import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Callable

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)

from loguru import logger

from src.core.evaluator import Evaluator
from src.core.trainer import TrainerExecutionPipeline
from src.data.data_eda import DataEDA
from src.visualization import MetricsPlotter
from src.config import load_config
from src.data.dataset_loaders import MetadataLoaderFactory
from src.utils.file_loader import CaseInsensitiveAssetResolver
from src.utils.checkpointing import select_best_checkpoint
from src.utils.logging_utils import log_prepare, time_it


def _deploy_best_checkpoint(cfg: Any, dataset_name: str, base_fn: str) -> Path:
    dataset_slug = cfg.paths.dataset_slug(dataset_name)
    task_type = (getattr(cfg, "task_type", None) or cfg.dataset.task_type).lower()
    if task_type not in {"iqa", "vqa"}:
        raise ValueError(f"Unsupported task_type for deployment: {task_type}")

    model_outputs_dir = cfg.paths.model_outputs_dir(dataset_name)
    monitor = cfg.train.checkpoint.monitor.lower()
    mode = cfg.train.checkpoint.mode.lower()
    best_checkpoint = select_best_checkpoint(
        model_outputs_dir=model_outputs_dir,
        base_fn=base_fn,
        monitor=monitor,
        mode=mode,
        secondary_monitor=cfg.train.checkpoint.secondary_monitor.lower(),
    )

    deploy_dir = cfg.paths.deploy_iqa_dir() if task_type == "iqa" else cfg.paths.deploy_vqa_dir()
    deploy_dir.mkdir(parents=True, exist_ok=True)

    deployed_path = deploy_dir / f"{dataset_slug}_best.pt"
    shutil.copy2(best_checkpoint, deployed_path)
    logger.info(f"Deployed best checkpoint: {best_checkpoint} -> {deployed_path}")
    return deployed_path


def deploy_best_checkpoint_after_run(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if result is False:
            logger.warning("Main run did not complete successfully; skipping checkpoint deployment.")
            return result

        if not isinstance(result, dict):
            logger.warning("Main run did not return deployment context; skipping checkpoint deployment.")
            return result

        cfg = result.get("cfg")
        dataset_name = result.get("dataset_name")
        base_fn = result.get("base_fn")
        if cfg is None or dataset_name is None or base_fn is None:
            logger.warning("Deployment context is incomplete; skipping checkpoint deployment.")
            return result

        _deploy_best_checkpoint(
            cfg=cfg,
            dataset_name=str(dataset_name),
            base_fn=str(base_fn),
        )
        return result

    return wrapper


def _plot_training_outputs(
    plotter: MetricsPlotter,
    train_logs_dir: Path,
    base_fn: str,
    dataset_name: str,
    n_splits: int,
    fast_run: bool = False,
) -> None:
    model_metrics = {}
    max_fold = 1 if fast_run else n_splits

    for fold_idx in range(1, max_fold + 1):
        fold_base = f"{base_fn}_fold{fold_idx}"
        history_path = train_logs_dir / f"{fold_base}_history.csv"
        manifest_path = train_logs_dir / f"{fold_base}_manifest.csv"

        if history_path.exists():
            fold_key = f"{plotter.model_name}_Fold{fold_idx}"
            version = f"fold{fold_idx}"
            model_metrics[fold_key] = {
                "version": version,
                "fold_label": f"Fold{fold_idx:02d}",
                "csv_path": history_path,
            }
            plotter.render_registered("training_history", csv_path=history_path, version=version)
        else:
            logger.warning(f"History file missing for fold {fold_idx}: {history_path}")

        if manifest_path.exists():
            plotter.render_registered("residuals", csv_path=manifest_path, version=f"fold{fold_idx}")
            plotter.render_registered("error_by_mos_bin", csv_path=manifest_path, version=f"fold{fold_idx}")
        else:
            logger.warning(f"Manifest file missing for fold {fold_idx}: {manifest_path}")

    if model_metrics:
        plotter.render_registered("comparison", metrics_csv_dict=model_metrics, dataset_name=dataset_name)
        plotter.render_registered("fold_summary", metrics_csv_dict=model_metrics, dataset_name=dataset_name)
        logger.info("Training visualizations generated successfully.")
    else:
        logger.warning(f"No valid training history found in {train_logs_dir}.")


@deploy_best_checkpoint_after_run
@time_it
def main() -> dict[str, Any] | bool:
    parser = argparse.ArgumentParser(description="Deep VQA/IQA General Data-Driven Framework")
    parser.add_argument("--model", type=str, default="swin_iqa")
    parser.add_argument("--dataset", type=str, default="TID2013")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--skip_integrity",
        action="store_true",
        help="Skip media integrity checks. Not recommended for training.",
    )
    args = parser.parse_args()

    requested_dataset = args.dataset
    dataset_name = MetadataLoaderFactory.normalize_key(requested_dataset)
    model_name = args.model.lower()

    cfg = load_config(
        config_dir=Path("train-config"),
        model_name=model_name,
        dataset_name=dataset_name,
    )

    logger.info(
        f"CLI: model={model_name}, backbone={cfg.model.backbone} -> "
        f"task_type={cfg.task_type}, dataset={cfg.dataset.name} ({dataset_name})"
    )

    data_dir = cfg.paths.resolve(Path(cfg.dataset.paths.root) / cfg.dataset.paths.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    train_logs_dir = cfg.paths.train_logs_dir(dataset_name)
    plots_dir = cfg.paths.plots_dir(dataset_name)
    model_outputs_dir = cfg.paths.model_outputs_dir(dataset_name)

    train_logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        logger.warning("Smoke test mode activated.")
        cfg.train.epochs = 1
        cfg.preprocessing.batch_size = 2
        cfg.preprocessing.k_fold = 2
        cfg.train.early_stop.enabled = False

    base_fn = log_prepare(
        cfg=cfg,
        model_name=cfg.model.name,
        dataset_name=dataset_name,
    )

    eda_engine = DataEDA(
        cfg=cfg,
        dataset_name=dataset_name,
        data_dir=data_dir,
    )

    eda_engine.run_full_eda(skip_integrity=args.skip_integrity)

    if eda_engine.df is None or "split" not in eda_engine.df.columns:
        logger.error("EDA pipeline failure.")
        return False

    if "mos_min" in eda_engine.stats and "mos_max" in eda_engine.stats:
        logger.info(f"MOS range: [{eda_engine.stats['mos_min']:.3f}, {eda_engine.stats['mos_max']:.3f}]")

    if args.smoke_test and eda_engine.df is not None:
        eda_engine.df = eda_engine.df.sample(
            n=min(16, len(eda_engine.df)), random_state=cfg.preprocessing.seed
        ).reset_index(drop=True)

    if "split" not in eda_engine.df.columns:
        logger.error("Data EDA did not create split column!")
        return False

    resolver = CaseInsensitiveAssetResolver(target_dir=data_dir)

    pipeline = TrainerExecutionPipeline(
        cfg=cfg,
        eda_df=eda_engine.df,
        data_dir=data_dir,
        resolver=resolver,
        fast_run=args.smoke_test,
    )

    evaluator = Evaluator(
        cfg=cfg,
        logs_dir=train_logs_dir,
    )
    evaluator.base_filename = base_fn

    try:
        completed_folds = pipeline.execute_cross_validation(evaluator=evaluator)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

    plotter = MetricsPlotter(
        cfg=cfg,
        plots_dir=plots_dir,
    )

    try:
        _plot_training_outputs(
            plotter=plotter,
            train_logs_dir=train_logs_dir,
            base_fn=base_fn,
            dataset_name=dataset_name,
            n_splits=completed_folds,
            fast_run=args.smoke_test,
        )
    except Exception as e:
        # Checkpoint deployment is still valid if an optional visualization fails.
        logger.exception(f"Training visualization failed; continuing to deployment: {e}")
    logger.info("System lifecycle completed successfully.")
    return {"cfg": cfg, "dataset_name": dataset_name, "base_fn": base_fn}


if __name__ == "__main__":
    main()
