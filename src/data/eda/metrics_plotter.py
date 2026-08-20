"""Backward-compatible import path for the moved training plotter."""

from src.visualization.training_plots import MetricsPlotter, PLOT_REGISTRY

__all__ = ["MetricsPlotter", "PLOT_REGISTRY"]
