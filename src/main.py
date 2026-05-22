import os
# src/__init__.py 会在导入 src 包时执行，但 main.py 可能先执行
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)  # 或 cv2.setNumThreads(1)

import argparse
import yaml
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional
from collections.abc import Mapping

from src.utils.logging_utils import log_prepare, time_it
from src.data.data_eda import DataEDA
from src.core.evaluator import Evaluator
from src.utils.path_manager import PathManager
from src.utils.file_loader import CaseInsensitiveAssetResolver
from src.data.eda.metrics_plotter import MetricsPlotter
from src.core.trainer import TrainerExecutionPipeline
from src.utils.config_loader import load_system_config
import pdb



@time_it
def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="Deep VQA/IQA General Data-Driven Framework")
    parser.add_argument('-c', '--config', type=str, default='basic', help="Basic config filename")
    parser.add_argument('--model', type=str, default='resnet_iqa', help="Model filename (e.g., resnet_iqa)")
    parser.add_argument('--dataset', type=str, default='TID2013', help="Target dataset name")
    parser.add_argument('--smoke_test', action='store_true', help="Activate fast smoke check")
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.model = args.model.lower()

    config = load_system_config(args.config, args.dataset)
    config['dataset_name'] = args.dataset

    logger.debug(f"[Main] 命令行参数: config={args.config}, model={args.model}, dataset={args.dataset}, smoke_test={args.smoke_test}")

    dataset_name = config['dataset_name']
    task_type = config.get('task_type', 'vqa')
    model_name = config['model'].get('name', 'default_model')

    logger.debug(f"[Main] 任务配置: dataset={dataset_name}, task_type={task_type}, model={model_name}")



    try:
        data_dir = PathManager.get_dataset_dir(dataset_name).resolve()

        # 调试锚点：校验数据集物理路径
        # Help: 在这里输入 'p data_dir' 检查路径解析是否正确
        # Help: 如果发现路径偏移，检查 PathManager 的根目录设置
        if not data_dir.exists():
            raise FileNotFoundError(f"数据集物理路径不存在: {data_dir}")
        else:
            # 检查是否因为目录嵌套导致路径配偏了
            sub_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
            if len(sub_dirs) == 1 and sub_dirs[0].name.lower() == dataset_name.lower():
                logger.warning(f"⚠️ 发现目录嵌套，自动下潜: {sub_dirs[0]}")
                data_dir = sub_dirs[0].resolve()
                logger.debug(f"[PathManager] 修正后路径: {data_dir}")

    except FileNotFoundError as e:
        logger.error(f"❌ 数据集路径错误: {e}")
        # Help: 在这里输入 'p dataset_name' 查看数据集名称
        # Help: 输入 'p data_dir' 查看当前尝试的路径
        # Help: 检查 PathManager 的根目录配置是否正确
        pdb.set_trace()
        pdb.set_trace()
        return



    train_logs_dir = PathManager.get_results_dir(dataset_name, "train_logs")
    plots_dir = PathManager.get_results_dir(dataset_name, "plots")
    model_outputs_dir = PathManager.get_results_dir(dataset_name, "model_outputs")

    # 确保目录存在
    train_logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"[Main] 结果目录准备: train_logs={train_logs_dir}, plots={plots_dir}, outputs={model_outputs_dir}")

    config.setdefault('logging', {}).update({
        'save_dir': str(model_outputs_dir),
        'log_dir': str(train_logs_dir)
    })



    # 3. 冒烟测试模式设置
    if args.smoke_test:
        logger.warning("[Main] Smoke test mode activated.")
        config.setdefault('train', {}).update({'epochs': 1, 'fast_run': True})
        config.setdefault('preprocessing', {}).update({'batch_size': 2, 'k_fold': 2})
        if 'early_stop' in config.get('train', {}):
            config['train']['early_stop']['enabled'] = False



    # 4. 初始化核心组件
    base_fn = log_prepare(model_name=model_name, dataset_name=dataset_name)
    eda_engine = DataEDA(
        dataset_name=args.dataset,
        data_dir=data_dir,
        dataset_info=config['dataset_info']
    )


    eda_engine.run_full_eda(skip_integrity=True)
    if eda_engine.df is None or 'split' not in eda_engine.df.columns:
        logger.error("[Main] EDA pipeline failure.")
        # Help: 检查EDA内部状态，输入 'p eda_engine.df' 查看数据框
        pdb.set_trace()
        return

    if args.smoke_test and eda_engine.df is not None:
        logger.info(f"⚡ [Main] Smoke sampling: Truncating metadata from {len(eda_engine.df)} to 16 samples.")
        eda_engine.df = eda_engine.df.sample(
            n=min(16, len(eda_engine.df)),
            random_state=42
        ).reset_index(drop=True)


    if 'split' not in eda_engine.df.columns:
        logger.error("[Main] 严重错误: 数据EDA未正确划分 split 列，无法安全隔离测试集！")
        # Help: 查看数据框列名，输入 'p eda_engine.df.columns.tolist()'
        pdb.set_trace()
        return

    resolver = CaseInsensitiveAssetResolver(
        target_dir=data_dir,
        allowed_extensions=eda_engine.file_extensions
    )

    # --- 诊断插入点 ---
    logger.info(f"DEBUG: 最终确定的物理数据锚点: {data_dir}")
    resolver_debug = resolver
    # 随机取一个 metadata 里的 ID 进行测试
    sample_test = eda_engine.df['sample_id'].iloc[0]
    try:
        path_test = resolver_debug.resolve(sample_test)
        logger.info(f"DEBUG: 测试解析成功! 物理路径为: {path_test}")
    except Exception as e:
        logger.error(f"DEBUG: 测试解析失败! sample_id: {sample_test}, 错误: {e}")
        # Help: 检查文件实际是否存在，输入 'import os; os.listdir(data_dir)[:10]'
        pdb.set_trace()
    # ------------------

    # 5. 训练流水线
    pipeline = TrainerExecutionPipeline(
        config=config,
        eda_df=eda_engine.df,
        data_dir=data_dir,
        resolver=resolver
    )

    evaluator = Evaluator(task_type=task_type, model_name=model_name)
    evaluator.base_filename = base_fn
    try:
        pipeline.execute_cross_validation(evaluator=evaluator)  # FIXME
    except Exception as e:
        logger.error(f"[Main] 训练执行失败: {e}")
        # Help: 使用 'p e' 查看异常详情
        # Help: 输入 'p config.get("train", {})' 查看训练配置
        pdb.set_trace()
        return



    # 6. 可视化分析
    plotter = MetricsPlotter(task_type=task_type,
                             model_name=model_name,
                             plots_dir=plots_dir)
    arena_payload = {}
    n_splits = config.get('preprocessing', {}).get('k_fold', 5)


    for fold_idx in range(1, n_splits + 1):
        fold_csv_path = train_logs_dir / f"{base_fn}_fold{fold_idx}_history.csv"

        if fold_csv_path.exists():
            fold_key = f"{model_name}_Fold{fold_idx}"
            arena_payload[fold_key] = {"version": "v1",
                                       "csv_path": fold_csv_path}
            if fold_idx == 1:
                plotter.plot_training_history(csv_path=fold_csv_path,
                                              version="v1")
                plotter.plot_residuals(csv_path=fold_csv_path,
                                       version="v1")

        if config.get('train', {}).get('fast_run', True) and fold_idx == 1:
            break

    if eda_engine.df is not None and not eda_engine.df.empty:
        logger.info(f"[Main] Generating traditional metrics distribution...")
        plotter.plot_traditional_metrics_distribution(data_df=eda_engine.df, dataset_name=dataset_name)



    # 7. 竞技场对比与收尾
    if arena_payload:
        plotter.plot_comparison(metrics_csv_dict=arena_payload, dataset_name=dataset_name)
        logger.info("[Main] System lifecycle completed successfully.")
    else:
        logger.warning(f"[Main] No valid training history found in {train_logs_dir}.")


if __name__ == "__main__":
    main()