# src/data/eda/metrics_plotter.py
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.config.schemas import Config


class MetricsPlotter:
    """Plot training curves, residuals, and model comparison charts."""

    def __init__(
        self,
        cfg: Config,
        plots_dir: Path,
        version: str = "v1",
    ):
        """
        Initialize MetricsPlotter.

        Args:
            cfg: Pydantic configuration object
            plots_dir: Directory to save plots
            version: Version string for file naming
        """
        self.cfg = cfg
        self.task_type = cfg.task_type
        self.model_name = cfg.model.name
        self.version = version
        self.current_date = time.strftime("%Y%m%d")

        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        plt.switch_backend("Agg")
        sns.set_style("whitegrid")

        self.base_prefix = f"{self.current_date}_{self.task_type}_{self.model_name}"

    def plot_training_history(self, csv_path: Path, version: str = "v1") -> None:
        """Plot training history: loss, PLCC/SROCC, RMSE/R2, KROCC."""
        if not csv_path.exists():
            logger.warning(f"History file not found: {csv_path}")
            raise FileNotFoundError(f"Missing history logs at {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"CSV file is empty: {csv_path}")
                return
        except Exception as e:
            logger.error(f"Failed to read CSV: {csv_path}, error: {e}")
            raise

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss Curve
        if "train_loss" in df.columns and "val_loss" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="train_loss",
                ax=axes[0, 0],
                label="Train Loss",
                marker="o",
                markersize=4,
                linewidth=2,
                color="tab:blue",
            )
            sns.lineplot(
                data=df,
                x="epoch",
                y="val_loss",
                ax=axes[0, 0],
                label="Val Loss",
                marker="s",
                markersize=4,
                linewidth=2,
                color="tab:orange",
            )
            axes[0, 0].set_title("Loss Convergence Curve", fontsize=12, fontweight="bold")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()

        # 2. PLCC and SROCC
        if "plcc" in df.columns and "srocc" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="plcc",
                ax=axes[0, 1],
                label="PLCC (Pearson)",
                marker="o",
                markersize=4,
                color="forestgreen",
                linewidth=2,
            )
            sns.lineplot(
                data=df,
                x="epoch",
                y="srocc",
                ax=axes[0, 1],
                label="SROCC (Spearman)",
                marker="s",
                markersize=4,
                color="royalblue",
                linewidth=2,
            )
            axes[0, 1].set_title("PLCC / SROCC vs Human MOS", fontsize=12, fontweight="bold")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Correlation")
            axes[0, 1].legend()
            axes[0, 1].set_ylim(-0.05, 1.05)
        else:
            axes[0, 1].text(0.5, 0.5, "PLCC/SROCC Data Missing", ha="center", va="center", color="gray")
            axes[0, 1].axis("off")

        # 3. RMSE and R2
        if "rmse" in df.columns and "r2" in df.columns:
            ax1 = axes[1, 0]
            sns.lineplot(
                data=df,
                x="epoch",
                y="rmse",
                ax=ax1,
                label="RMSE",
                marker="o",
                markersize=4,
                color="crimson",
                linewidth=2,
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("RMSE", color="crimson")
            ax1.tick_params(axis="y", labelcolor="crimson")
            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()
            sns.lineplot(
                data=df,
                x="epoch",
                y="r2",
                ax=ax2,
                label="R²",
                marker="s",
                markersize=4,
                color="darkorange",
                linewidth=2,
            )
            ax2.set_ylabel("R²", color="darkorange")
            ax2.tick_params(axis="y", labelcolor="darkorange")
            ax2.legend(loc="upper right")
            axes[1, 0].set_title("RMSE & R²", fontsize=12, fontweight="bold")
        else:
            axes[1, 0].text(0.5, 0.5, "RMSE/R2 Data Missing", ha="center", va="center", color="gray")
            axes[1, 0].axis("off")

        # 4. KROCC
        if "krocc" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="krocc",
                ax=axes[1, 1],
                label="KROCC (Kendall)",
                marker="^",
                markersize=4,
                color="purple",
                linewidth=2,
            )
            axes[1, 1].set_title("KROCC (Kendall Tau)", fontsize=12, fontweight="bold")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("KROCC")
            axes[1, 1].set_ylim(-0.05, 1.05)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, "KROCC Data Missing", ha="center", va="center", color="gray")
            axes[1, 1].axis("off")

        plt.suptitle(
            f"Training History | Model: {self.model_name} ({version})",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.base_prefix}_{version}_training_history.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")

    def plot_residuals(self, csv_path: Path, version: str = "v1") -> None:
        """Plot residual diagnostics: scatter, alignment, and distribution."""
        if not csv_path.exists():
            logger.warning(f"File not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"CSV file is empty: {csv_path}")
                return
        except Exception as e:
            logger.error(f"Failed to read CSV: {csv_path}, error: {e}")
            return

        if "pred" not in df.columns or "true" not in df.columns:
            logger.warning(f"CSV missing 'pred' or 'true' columns: {csv_path}")
            return

        y_true = np.asarray(df["true"], dtype=np.float64)
        y_pred = np.asarray(df["pred"], dtype=np.float64)
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Residual scatter
        axes[0].scatter(
            y_pred,
            residuals,
            alpha=0.5,
            s=20,
            color="dodgerblue",
            edgecolors="w",
            linewidths=0.3,
        )
        axes[0].axhline(y=0, color="crimson", linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Predicted MOS")
        axes[0].set_ylabel("Residual (True - Pred)")
        axes[0].set_title("Residuals vs Predicted", fontsize=11, fontweight="bold")

        # 2. True vs Predicted
        axes[1].scatter(
            y_true,
            y_pred,
            alpha=0.5,
            s=20,
            color="darkgreen",
            edgecolors="w",
            linewidths=0.3,
        )
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot(
            [min_val, max_val],
            [min_val, max_val],
            "crimson",
            linestyle="--",
            linewidth=1.5,
            label="Perfect",
        )
        axes[1].set_xlabel("True MOS")
        axes[1].set_ylabel("Predicted MOS")
        axes[1].set_title("Predicted vs True", fontsize=11, fontweight="bold")
        axes[1].legend()

        # 3. Residual distribution
        sns.histplot(
            residuals,
            kde=True,
            ax=axes[2],
            color="purple",
            edgecolor="black",
            alpha=0.6,
            bins=20,
        )
        axes[2].axvline(x=0, color="crimson", linestyle="--", linewidth=1.5)
        axes[2].set_xlabel("Residual")
        axes[2].set_ylabel("Density")
        axes[2].set_title("Residual Distribution", fontsize=11, fontweight="bold")

        plt.suptitle(f"Residual Analysis | Model: {self.model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.base_prefix}_{version}_residuals.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")

    def plot_comparison(self, metrics_csv_dict: dict, dataset_name: str) -> None:
        """Bar chart comparing PLCC, SROCC, RMSE across models."""
        if not metrics_csv_dict:
            logger.warning("No metrics files provided for comparison.")
            return

        models = []
        plcc_list, srocc_list, rmse_list = [], [], []

        for m_name, info in metrics_csv_dict.items():
            csv_path = info.get("csv_path")
            if not csv_path or not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
            except Exception:
                continue

            last_row = df.iloc[-1]
            models.append(f"{m_name}\n({info.get('version', 'v1')})")
            plcc_list.append(last_row.get("plcc", 0))
            srocc_list.append(last_row.get("srocc", 0))
            rmse_list.append(last_row.get("rmse", 0))

        if not models:
            logger.warning("No valid metrics loaded.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_payload = [
            {"data": plcc_list, "title": "PLCC (higher is better)", "color": "seagreen", "ax": axes[0]},
            {"data": srocc_list, "title": "SROCC (higher is better)", "color": "royalblue", "ax": axes[1]},
            {"data": rmse_list, "title": "RMSE (lower is better)", "color": "crimson", "ax": axes[2]},
        ]

        for payload in metrics_payload:
            ax = payload["ax"]
            bars = ax.bar(
                models,
                payload["data"],
                color=payload["color"],
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_title(payload["title"], fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

            if "RMSE" in payload["title"]:
                ax.set_ylim(0, max(payload["data"]) * 1.2)
            else:
                ax.set_ylim(0, 1.05)

            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + (h * 0.01),
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="semibold",
                )

        plt.suptitle(f"Model Comparison on {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / f"comparison_{dataset_name}_{self.current_date}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")