# src/models/metrics.py
from typing import Dict

import torch


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute PLCC, SROCC, RMSE, MAE."""
    import numpy as np
    from scipy.stats import spearmanr

    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    plcc = (pred_n * target_n).sum() / (
        np.sqrt((pred_n**2).sum()) * np.sqrt((target_n**2).sum()) + 1e-8
    )

    srocc, _ = spearmanr(pred, target)
    rmse = np.sqrt(((pred - target) ** 2).mean())
    mae = np.abs(pred - target).mean()

    return {"plcc": float(plcc), "srocc": float(srocc), "rmse": float(rmse), "mae": float(mae)}


__all__ = ["compute_metrics"]
