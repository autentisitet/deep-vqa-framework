# src/visualization/training_plots.py
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.config.schemas import Config
from src.utils.registry import Registry


PLOT_REGISTRY = Registry[dict]("visualization_plots")
PLOT_REGISTRY.register(
    "training_history",
    {"method": "plot_training_history", "group": "training", "enabled": True},
)
PLOT_REGISTRY.register(
    "residuals",
    {"method": "plot_residuals", "group": "evaluation", "enabled": True},
)
PLOT_REGISTRY.register(
    "comparison",
    {"method": "plot_comparison", "group": "evaluation", "enabled": True},
)
PLOT_REGISTRY.register(
    "fold_summary",
    {"method": "plot_fold_summary", "group": "evaluation", "enabled": True},
)
PLOT_REGISTRY.register(
    "residuals_vs_true",
    {"method": "plot_residuals_vs_true", "group": "diagnostics", "enabled": True},
)
PLOT_REGISTRY.register(
    "error_by_mos_bin",
    {"method": "plot_error_by_mos_bin", "group": "diagnostics", "enabled": True},
)


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

    def _safe_dataset_name(self, dataset_name: str) -> str:
        return self.cfg.paths.dataset_slug(dataset_name)

    def render_registered(self, plot_name: str, **kwargs) -> None:
        """Render a registered plot by name without exposing method dispatch to callers."""
        spec = PLOT_REGISTRY.get(plot_name)
        if not spec.get("enabled", True):
            logger.info(f"Plot disabled: {plot_name}")
            return
        renderer = getattr(self, spec["method"], None)
        if renderer is None or not callable(renderer):
            raise AttributeError(f"Registered plot renderer is unavailable: {spec['method']}")
        renderer(**kwargs)

    def render_group(self, group: str, **context) -> None:
        """Render all enabled plots in a registered group."""
        for plot_name, spec in PLOT_REGISTRY.items():
            if spec.get("group") == group and spec.get("enabled", True):
                self.render_registered(plot_name, **context.get(plot_name, {}))

    @staticmethod
    def _read_csv(csv_path: Path, *, required_columns: tuple[str, ...] = ()) -> pd.DataFrame | None:
        """Read a metrics CSV and return None when it is missing or unusable."""
        if not csv_path.exists():
            logger.warning(f"Metrics file not found: {csv_path}")
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error(f"Failed to read metrics file: {csv_path} | {exc}")
            return None
        if df.empty:
            logger.warning(f"Metrics file is empty: {csv_path}")
            return None
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            logger.warning(f"Metrics file missing columns {missing}: {csv_path}")
            return None
        return df

    @staticmethod
    def _plot_lines(ax, df: pd.DataFrame, specs: list[dict]) -> None:
        """Plot multiple metric lines using one shared configuration helper."""
        for spec in specs:
            if spec["column"] not in df.columns:
                continue
            sns.lineplot(
                data=df,
                x="epoch",
                y=spec["column"],
                ax=ax,
                label=spec["label"],
                marker=spec.get("marker", "o"),
                markersize=spec.get("markersize", 4),
                linewidth=spec.get("linewidth", 2),
                color=spec["color"],
            )

    @staticmethod
    def _save_figure(fig, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_training_history(self, csv_path: Path, version: str = "v1") -> None:
        """Plot training history: loss, PLCC/SROCC, RMSE/R2, KROCC."""
        df = self._read_csv(csv_path, required_columns=("epoch",))
        if df is None:
            raise FileNotFoundError(f"Missing or invalid history logs at {csv_path}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss Curve
        if "train_loss" in df.columns and "val_loss" in df.columns:
            self._plot_lines(axes[0, 0], df, [
                {"column": "train_loss", "label": "Train Loss", "marker": "o", "color": "tab:blue"},
                {"column": "val_loss", "label": "Val Loss", "marker": "s", "color": "tab:orange"},
            ])
            axes[0, 0].set_title("Loss Convergence Curve", fontsize=12, fontweight="bold")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()

        # 2. PLCC and SROCC
        if "plcc" in df.columns and "srocc" in df.columns:
            self._plot_lines(axes[0, 1], df, [
                {"column": "plcc", "label": "PLCC (Pearson)", "marker": "o", "color": "forestgreen"},
                {"column": "srocc", "label": "SROCC (Spearman)", "marker": "s", "color": "royalblue"},
            ])
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
            self._plot_lines(axes[1, 1], df, [
                {"column": "krocc", "label": "KROCC (Kendall)", "marker": "^", "color": "purple"},
            ])
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
        save_path = self.plots_dir / f"{self.base_prefix}_{version}_training_history.png"
        self._save_figure(fig, save_path)

    def plot_residuals(self, csv_path: Path, version: str = "v1") -> None:
        """Plot residual diagnostics: scatter, alignment, and distribution."""
        if not csv_path.exists():
            logger.warning(f"File not found: {csv_path}")
            return

        df = self._read_csv(csv_path)
        if df is None:
            return

        true_col = "true_mos" if "true_mos" in df.columns else "true"
        pred_col = "pred_mos" if "pred_mos" in df.columns else "pred"
        if true_col not in df or pred_col not in df:
            logger.warning(f"Manifest missing prediction columns: {csv_path}")
            return
        y_true = np.asarray(df[true_col], dtype=np.float64)
        y_pred = np.asarray(df[pred_col], dtype=np.float64)
        residuals = y_pred - y_true

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
        axes[0].set_ylabel("Residual (Pred - True)")
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
        span = max(max_val - min_val, 1e-6)
        line_min = min_val - span * 0.03
        line_max = max_val + span * 0.03
        axes[1].plot(
            [line_min, line_max],
            [line_min, line_max],
            "crimson",
            linestyle="--",
            linewidth=1.5,
            label="Perfect",
        )
        axes[1].set_xlim(line_min, line_max)
        axes[1].set_ylim(line_min, line_max)
        axes[1].set_aspect("equal", adjustable="box")
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
        save_path = self.plots_dir / f"{self.base_prefix}_{version}_residuals.png"
        self._save_figure(fig, save_path)

        fig_true, ax_true = plt.subplots(figsize=(6, 5))
        ax_true.scatter(y_true, residuals, alpha=0.55, s=20, color="teal", edgecolors="w", linewidths=0.3)
        ax_true.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
        ax_true.set_xlabel("True MOS")
        ax_true.set_ylabel("Residual (Pred - True)")
        ax_true.set_title("Residuals vs True MOS", fontweight="bold")
        self._save_figure(fig_true, self.plots_dir / f"{self.base_prefix}_{version}_residuals_vs_true.png")

        # Additional diagnostics are exported as tabular data for downstream use.
        analysis_dir = self.plots_dir.parent / "analysis" / "errors"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        enriched = df.copy()
        enriched["signed_error"] = residuals
        enriched["absolute_error"] = np.abs(residuals)
        enriched["mos_bin"] = pd.cut(enriched[true_col], bins=5, include_lowest=True)
        enriched.nlargest(20, "absolute_error").to_csv(analysis_dir / f"top_error_samples_{version}.csv", index=False)
        grouped = enriched.groupby("mos_bin", observed=False)["absolute_error"].agg(["count", "mean", "median", "max"])
        grouped.to_csv(analysis_dir / f"error_by_mos_bin_{version}.csv")

    def plot_residuals_vs_true(self, csv_path: Path, version: str = "v1") -> None:
        """Render the dedicated residual-vs-true-MOS diagnostic."""
        self.plot_residuals(csv_path, version)

    def plot_error_by_mos_bin(self, csv_path: Path, version: str = "v1") -> None:
        """Render and export mean absolute error for true-MOS bins."""
        df = self._read_csv(csv_path)
        if df is None:
            return
        true_col = "true_mos" if "true_mos" in df.columns else "true"
        pred_col = "pred_mos" if "pred_mos" in df.columns else "pred"
        if true_col not in df or pred_col not in df:
            return
        work = df[[true_col, pred_col]].copy()
        work["absolute_error"] = (work[pred_col] - work[true_col]).abs()
        work["mos_bin"] = pd.cut(work[true_col], bins=5, include_lowest=True)
        summary = work.groupby("mos_bin", observed=False)["absolute_error"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        summary.plot.bar(ax=ax, color="steelblue")
        ax.set_xlabel("True MOS bin")
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Error by True MOS Bin", fontweight="bold")
        self._save_figure(fig, self.plots_dir / f"{self.base_prefix}_{version}_error_by_mos_bin.png")

    def plot_comparison(self, metrics_csv_dict: dict, dataset_name: str) -> None:
        """Bar chart comparing PLCC, SROCC, and RMSE across folds."""
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
            fold_label = info.get("fold_label")
            models.append(fold_label or f"Fold{len(models) + 1:02d}")
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

        dataset_slug = self._safe_dataset_name(dataset_name)
        plt.suptitle(f"Fold Comparison on {dataset_slug}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / f"comparison_{dataset_slug}_{self.current_date}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")

    def plot_fold_summary(self, metrics_csv_dict: dict, dataset_name: str) -> None:
        """Plot final fold metrics and metric stability across folds."""
        rows = []
        for fold_name, info in metrics_csv_dict.items():
            csv_path = info.get("csv_path")
            if not csv_path or not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
            except Exception as e:
                logger.warning(f"Failed to read metrics for fold summary: {csv_path} | {e}")
                continue

            last_row = df.iloc[-1]
            fold_index = len(rows) + 1
            rows.append(
                {
                    "fold": f"Fold{fold_index:02d}",
                    "plcc": float(last_row.get("plcc", 0.0)),
                    "srocc": float(last_row.get("srocc", 0.0)),
                    "krocc": float(last_row.get("krocc", 0.0)),
                    "rmse": float(last_row.get("rmse", 0.0)),
                    "mae": float(last_row.get("mae", 0.0)),
                }
            )

        if not rows:
            logger.warning("No valid fold metrics loaded for summary plot.")
            return

        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        corr_df = df.melt(
            id_vars="fold",
            value_vars=[c for c in ["plcc", "srocc", "krocc"] if c in df.columns],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(data=corr_df, x="fold", y="value", hue="metric", ax=axes[0])
        axes[0].set_title("Correlation Metrics by Fold", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("Correlation")
        axes[0].set_ylim(0, 1.05)
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].legend(title="")

        err_df = df.melt(
            id_vars="fold",
            value_vars=[c for c in ["rmse", "mae"] if c in df.columns],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(data=err_df, x="fold", y="value", hue="metric", ax=axes[1])
        axes[1].set_title("Error Metrics by Fold", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("Error")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(title="")

        dataset_slug = self._safe_dataset_name(dataset_name)
        plt.suptitle(f"Cross-Fold Summary on {dataset_slug}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / f"fold_summary_{dataset_slug}_{self.current_date}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")
