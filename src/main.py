import argparse
import os
import pdb
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)

from loguru import logger

from src.core.evaluator import Evaluator
from src.core.trainer import TrainerExecutionPipeline
from src.data.data_eda import DataEDA
from src.data.eda.metrics_plotter import MetricsPlotter
from src.config import load_config
from src.utils.file_loader import CaseInsensitiveAssetResolver
from src.utils.logging_utils import log_prepare, time_it


@time_it
def main() -> None:
    parser = argparse.ArgumentParser(description="Deep VQA/IQA General Data-Driven Framework")
    parser.add_argument("--model", type=str, default="resnet_iqa")
    parser.add_argument("--dataset", type=str, default="TID2013")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    model_name = args.model.lower()

    cfg = load_config(
        config_dir=Path("config"),
        model_name=model_name,
        dataset_name=dataset_name,
    )

    logger.info(
        f"CLI: model={model_name}, backbone={cfg.model.backbone} -> "
        f"task_type={cfg.task_type}, dataset={dataset_name}"
    )

    data_dir = Path(cfg.dataset.paths.root) / cfg.dataset.paths.data
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    train_logs_dir = cfg.paths.logs_dir / dataset_name / "train_logs"
    plots_dir = cfg.paths.logs_dir / dataset_name / "plots"
    model_outputs_dir = cfg.paths.results_dir / dataset_name / "model_outputs"

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

    eda_engine.run_full_eda(skip_integrity=True)

    if eda_engine.df is None or "split" not in eda_engine.df.columns:
        logger.error("EDA pipeline failure.")
        pdb.set_trace()
        return

    if "mos_min" in eda_engine.stats and "mos_max" in eda_engine.stats:
        logger.info(f"MOS range: [{eda_engine.stats['mos_min']:.3f}, {eda_engine.stats['mos_max']:.3f}]")

    if args.smoke_test and eda_engine.df is not None:
        eda_engine.df = eda_engine.df.sample(
            n=min(16, len(eda_engine.df)), random_state=42
        ).reset_index(drop=True)

    if "split" not in eda_engine.df.columns:
        logger.error("Data EDA did not create split column!")
        pdb.set_trace()
        return

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
        pipeline.execute_cross_validation(evaluator=evaluator)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        pdb.set_trace()
        return

    plotter = MetricsPlotter(
        cfg=cfg,
        plots_dir=plots_dir,
    )

    model_metrics = {}
    n_splits = cfg.preprocessing.k_fold

    for fold_idx in range(1, n_splits + 1):
        fold_csv_path = train_logs_dir / f"{base_fn}_fold{fold_idx}_history.csv"
        if fold_csv_path.exists():
            fold_key = f"{cfg.model.name}_Fold{fold_idx}"
            model_metrics[fold_key] = {"version": "v1", "csv_path": fold_csv_path}
            if fold_idx == 1:
                plotter.plot_training_history(csv_path=fold_csv_path, version="v1")
                plotter.plot_residuals(csv_path=fold_csv_path, version="v1")

        if args.smoke_test and fold_idx == 1:
            break

    if model_metrics:
        plotter.plot_comparison(metrics_csv_dict=model_metrics, dataset_name=dataset_name)
        logger.info("System lifecycle completed successfully.")
    else:
        logger.warning(f"No valid training history found in {train_logs_dir}.")


if __name__ == "__main__":
    main()