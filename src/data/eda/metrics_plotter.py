# src/data/eda/metrics_plotter.py
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
from typing import Optional
from loguru import logger


# Targets: (可视化)
#   1. 绘制loss曲线，残差图，每个epoch的 loss output，RMSE、R2 score
#   2. 绘制SSIM、VIF、DLM、VMAF、NIQE   <-- 🚀 独家新增：学术传统指标对账热力分布图/箱线图
#
# Notes:
#   1. 存储的img注意文件名不要乱七八糟，要保留生成时间、model的信息 <-- 🚀 已锁定年月日_task_model规范
#   2. 以py文件所在的文件路径为基准：
#      img文件存储到../results/plots/中   <-- 🚀 完美对齐你的 results/plots 目录


class MetricsPlotter:
    """
    可视化工具：现在通过依赖注入接收存储路径，具备极高的灵活性。
    """
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

    def __init__(self, task_type: str = "iqa", model_name: str = "resnet50", plots_dir: Path = None):
        """
        初始化画图插件，锁定全局命名规范
        """
        self.task_type = task_type
        self.model_name = model_name
        self.current_date = time.strftime("%Y%m%d")

        if plots_dir:
            self.plots_dir = Path(plots_dir)
        else:
            self.plots_dir = self._PROJECT_ROOT / "results" / "plots"

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        # 强制锁定画图后端与美化，防止服务器无 GUI 报错
        plt.switch_backend('Agg')
        sns.set_style('whitegrid')

        # 生成标准的前缀名
        self.base_prefix = f"{self.current_date}_{self.task_type}_{self.model_name}"



    def plot_training_history(self, csv_path: Path, version: str = "v1"):
        """
        从 CSV 读取训练历史并绘制 2x2 的全方位核心指标演进曲线
        """
        if not csv_path.exists():
            logger.warning(f"⚠️  History file not found: {csv_path}")
            raise FileNotFoundError(f"Missing history logs at {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"⚠️ CSV file is empty: {csv_path}")
                return
        except Exception as e:
            logger.error(f"❌ Failed to read CSV: {csv_path}, error: {e}")
            raise
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss 曲线 (Train vs Val)
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            sns.lineplot(data=df, x='epoch', y='train_loss', ax=axes[0, 0],
                         label='Train Loss', marker='o', markersize=4, linewidth=2, color='tab:blue')
            sns.lineplot(data=df, x='epoch', y='val_loss', ax=axes[0, 0],
                         label='Val Loss', marker='s', markersize=4, linewidth=2, color='tab:orange')
            axes[0, 0].set_title('Loss Convergence Curve', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()

        # 2. 学术质量评价核心指标：PLCC & SROCC 曲线
        if 'plcc' in df.columns and 'srocc' in df.columns:
            sns.lineplot(data=df, x='epoch', y='plcc', ax=axes[0, 1],
                         label='PLCC (Pearson)', marker='o', markersize=4, color='forestgreen', linewidth=2)
            sns.lineplot(data=df, x='epoch', y='srocc', ax=axes[0, 1],
                         label='SROCC (Spearman)', marker='s', markersize=4, color='royalblue', linewidth=2)
            axes[0, 1].set_title('Alignment with Human MOS (PLCC/SROCC)', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Correlation Coefficient')
            axes[0, 1].legend()

            # 💎【核心修复 1】：动态纵轴防御。若指标跌为负数，自适应拉低地平线，绝不发生折线断裂悬空
            min_corr = min(df['plcc'].min(), df['srocc'].min(), -0.05)
            axes[0, 1].set_ylim(min_corr - 0.05, 1.05)
        else:
            axes[0, 0].text(0.5, 0.5, "Loss Data Missing", ha='center', va='center', color='gray')
            axes[0, 0].set_title('Loss Convergence (Missing Data)')
            axes[0, 0].axis('off') # 明确关闭坐标轴，而不是放任空白


        # 3. 误差评估：RMSE (左轴) & R² Score (双胞胎右轴)
        if 'rmse' in df.columns and 'r2' in df.columns:
            ax1 = axes[1, 0]
            sns.lineplot(data=df, x='epoch', y='rmse', ax=ax1,
                         label='RMSE', marker='o', markersize=4, color='crimson', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('RMSE (Lower is better)', color='crimson')
            ax1.tick_params(axis='y', labelcolor='crimson')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            sns.lineplot(data=df, x='epoch', y='r2', ax=ax2,
                         label='R²', marker='s', markersize=4, color='darkorange', linewidth=2)
            ax2.set_ylabel('R² Score (Goodness of Fit)', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            ax2.legend(loc='upper right')
            axes[1, 0].set_title('Error & Prediction Accuracy', fontsize=12, fontweight='bold')

        # 4. 辅助指标：KROCC 曲线
        if 'krocc' in df.columns:
            sns.lineplot(data=df, x='epoch', y='krocc', ax=axes[1, 1],
                         label='KROCC (Kendall)', marker='^', markersize=4, color='purple', linewidth=2)
            axes[1, 1].set_title('Kendall Tau Rank Correlation', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('KROCC')
            min_krocc = min(df['krocc'].min(), -0.05)
            axes[1, 1].set_ylim(min_krocc - 0.05, 1.05)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, "KROCC Metric Not Cached",
                            ha='center', va='center', fontsize=12, color='gray')
            axes[1, 1].axis('off')

        plt.suptitle(f"Framework History | Model: {self.model_name} ({version})", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.base_prefix}_{version}_training_history.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved training history curve: {save_path}")



    def plot_residuals(self, csv_path: Path, version: str = "v1"):
        """
        💎【核心修复 2】：改造成 1x3 的诊断大底图，强势融入高斯残差密度分布直方图，打破密集盲区
        """
        if not csv_path.exists():
            logger.warning(f"⚠️  File not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"⚠️ CSV file is empty: {csv_path}")
                return
        except Exception as e:
            logger.error(f"❌ Failed to read CSV: {csv_path}, error: {e}")
            return

        if 'pred' not in df.columns or 'true' not in df.columns:
            logger.warning(f"⚠️  CSV missing 'pred' or 'true' columns: {csv_path}")
            return

        y_true = np.asarray(df['true'], dtype=np.float64)
        y_pred = np.asarray(df['pred'], dtype=np.float64)
        residuals = y_true - y_pred

        # 拓宽为 1x3 画布，第三张子图专门留给高斯误差对账
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 残差异方差散点检查
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color='dodgerblue', edgecolors='w', linewidths=0.3)
        axes[0].axhline(y=0, color='crimson', linestyle='--', linewidth=1.5)
        axes[0].set_xlabel('Predicted Quality Score (MOS)')
        axes[0].set_ylabel('Residuals (True - Pred)')
        axes[0].set_title('Residuals Scatter (Homoscedasticity)', fontsize=11, fontweight='bold')

        # 2. 真实值 vs 预测值 线性回归检查
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=20, color='darkgreen', edgecolors='w', linewidths=0.3)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'crimson', linestyle='--', linewidth=1.5, label='Perfect Alignment')
        axes[1].set_xlabel('True Human MOS')
        axes[1].set_ylabel('Predicted Network MOS')
        axes[1].set_title('Linearity Alignment (MOS Accuracy)', fontsize=11, fontweight='bold')
        axes[1].legend()

        # 3. 💥【新增核心子图】：残差一维边缘高斯概率密度分布
        sns.histplot(residuals, kde=True, ax=axes[2], color='purple', edgecolor='black', alpha=0.6, bins=20)
        axes[2].axvline(x=0, color='crimson', linestyle='--', linewidth=1.5)
        axes[2].set_xlabel('Residual Error Value')
        axes[2].set_ylabel('Density / Count')
        axes[2].set_title('Error Distribution (Normality Check)', fontsize=11, fontweight='bold')

        plt.suptitle(f"Residual & Linearity Diagnostics | Model: {self.model_name}", fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.base_prefix}_{version}_residuals.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved residual diagnostics plot: {save_path}")




    def plot_traditional_metrics_distribution(self, data_df: pd.DataFrame, dataset_name: str, score_column: str = "mos"):
        """
        💎【核心修复 3】：解除模糊因变量指代，强制显式绑定显式得分列名，防止网络混淆
        """
        target_metrics = ['ssim', 'vif', 'dlm', 'vmaf', 'niqe']
        available_metrics = []
        for m in target_metrics:
            # 尝试大小写不敏感匹配
            matches = [col for col in data_df.columns if col.lower() == m.lower()]
            if matches:
                available_metrics.append(matches[0])  # 用实际的列名

        if not available_metrics:
            logger.info(f"ℹ️  No traditional benchmarking metrics (SSIM/VIF/etc.) found in the current dataframe.")
            return


        # 稳固对账路由检查
        if score_column not in data_df.columns:
            # 尝试大小写不敏感匹配
            possible_cols = ['mos', 'true', 'score', 'target', 'label']
            matched = None
            for col in possible_cols:
                matches = [c for c in data_df.columns if c.lower() == col.lower()]
                if matches:
                    matched = matches[0]
                    break

            if matched:
                resolved_score = matched
                logger.warning(f"⚠️  Specified '{score_column}' not found. Using '{resolved_score}' instead.")
            else:
                # 如果都找不到，报错而不是静默用最后一列
                logger.error(f"❌ 找不到评分列。可用的列: {data_df.columns.tolist()}")
                return
        else:
            resolved_score = score_column


        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5), squeeze=False)

        for i, metric in enumerate(available_metrics):
            sns.regplot(data=data_df, x=metric, y=resolved_score, ax=axes[0, i],
                        scatter_kws={'alpha':0.4, 's':15, 'color':'purple'},
                        line_kws={'color':'crimson', 'linestyle':'-.'})
            axes[0, i].set_title(f'{metric.upper()} vs {resolved_score.upper()}', fontsize=11, fontweight='bold')
            axes[0, i].set_xlabel(metric.upper())
            axes[0, i].set_ylabel(resolved_score.upper())

        plt.suptitle(f"Traditional Academic Benchmarks Distribution on [{dataset_name}]", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.current_date}_{dataset_name}_traditional_benchmarks.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved traditional benchmarks distribution report: {save_path}")




    def plot_comparison(self, metrics_csv_dict: dict, dataset_name: str):
        """
        横向大PK竞技场：横向对比多个不同算法/模型的 PLCC, SROCC 和 RMSE 指标
        """
        if not metrics_csv_dict:
            logger.warning("⚠️  No metrics files provided for cross-model benchmarking.")
            return

        models = []
        plcc_list, srocc_list, rmse_list = [], [], []

        for m_name, info in metrics_csv_dict.items():
            csv_path = info.get('csv_path')
            if not csv_path or not csv_path.exists():
                logger.debug(f"CSV not found: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    logger.warning(f"Empty CSV: {csv_path}")
                    continue
            except Exception as e:
                logger.error(f"Failed to read CSV {csv_path}: {e}")
                continue

            last_row = df.iloc[-1]
            models.append(f"{m_name}\n({info.get('version', 'v1')})")
            plcc_list.append(last_row.get('plcc', 0))
            srocc_list.append(last_row.get('srocc', 0))
            rmse_list.append(last_row.get('rmse', 0))

        if not models:
            logger.warning("⚠️  No valid metrics rows successfully loaded.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_payload = [
            {"data": plcc_list, "title": "PLCC Comparison (Higher is better)", "color": "seagreen", "ax": axes[0]},
            {"data": srocc_list, "title": "SROCC Comparison (Higher is better)", "color": "royalblue", "ax": axes[1]},
            {"data": rmse_list, "title": "RMSE Comparison (Lower is better)", "color": "crimson", "ax": axes[2]}
        ]

        for payload in metrics_payload:
            ax = payload["ax"]
            # 1. 执行绘图
            bars = ax.bar(models, payload["data"], color=payload["color"], alpha=0.75, edgecolor='black', linewidth=0.5)
            ax.set_title(payload["title"], fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=15)

            # 2. 💎【关键修复】：强行约束坐标轴
            # 这里检查当前 payload 的标题来判断是哪种指标
            if "RMSE" in payload["title"]:
                # RMSE 越小越好，从 0 到“最大值+20%余量”，保证视觉上的公平性
                ax.set_ylim(0, max(payload["data"]) * 1.2)
            else:
                # PLCC 和 SROCC 是相关性，必须展示 [0, 1] 的满额度，防止小波动被放大
                ax.set_ylim(0, 1.05)

            # 3. 绘制数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height + (height * 0.01),
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='semibold')

        plt.suptitle(f"Cross-Model Benchmark Arena on Dataset: {dataset_name}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / f"comparison_arena_{dataset_name}_{self.current_date}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved tournament comparison leaderboard: {save_path}")