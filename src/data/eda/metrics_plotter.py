"""Backward-compatible import path for the moved training plotter."""

from src.visualization.training_plots import PLOT_REGISTRY, MetricsPlotter

__all__ = ["MetricsPlotter", "PLOT_REGISTRY"]
