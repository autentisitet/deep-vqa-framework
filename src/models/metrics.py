"""Quality-assessment metric calculations and optional nonlinear score mapping."""

from __future__ import annotations

from collections.abc import Callable
from typing import NotRequired, TypedDict

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr


class MappingDetails(TypedDict):
    applied: bool
    method: str
    parameters: NotRequired[list[float]]
    error: NotRequired[str]


class MappedMetrics(TypedDict):
    raw: dict[str, float]
    mapped: dict[str, float] | None
    mapping: MappingDetails


def _to_finite_float_array(value: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    """Convert a prediction or target into a finite 1-D float array."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Metric inputs must contain only finite values")
    return values


def _safe_correlation(
    function: Callable[[np.ndarray, np.ndarray], tuple[float, float]], left: np.ndarray, right: np.ndarray
) -> float:
    """Return a correlation, guarding degenerate inputs with ``0.0``."""
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = function(left, right)[0]
    return float(np.nan_to_num(value))


def compute_metrics(
    pred: torch.Tensor | np.ndarray | list[float],
    target: torch.Tensor | np.ndarray | list[float],
) -> dict[str, float]:
    """Compute raw PLCC, SROCC, KROCC, RMSE, and MAE without score fitting."""
    pred_values = _to_finite_float_array(pred)
    target_values = _to_finite_float_array(target)
    if len(pred_values) != len(target_values):
        raise ValueError("Prediction and target lengths must match")
    if not len(pred_values):
        return {"plcc": 0.0, "srocc": 0.0, "krocc": 0.0, "rmse": 0.0, "mae": 0.0}

    residual = pred_values - target_values
    return {
        "plcc": _safe_correlation(pearsonr, pred_values, target_values),
        "srocc": _safe_correlation(spearmanr, pred_values, target_values),
        "krocc": _safe_correlation(kendalltau, pred_values, target_values),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def five_parameter_logistic(
    scores: np.ndarray,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
    beta5: float,
) -> np.ndarray:
    """Map objective scores with the common five-parameter VQEG/LIVE function."""
    exponent = np.clip(beta2 * (scores - beta3), -60.0, 60.0)
    return beta1 * (0.5 - 1.0 / (1.0 + np.exp(exponent))) + beta4 * scores + beta5


def fit_five_parameter_logistic(
    pred: torch.Tensor | np.ndarray | list[float],
    target: torch.Tensor | np.ndarray | list[float],
) -> tuple[np.ndarray, tuple[float, float, float, float, float]]:
    """Fit and apply the five-parameter mapping on one evaluation set.

    This is a post-hoc reporting transform. It must not be used for checkpoint
    selection because its parameters are fitted to the same target MOS values
    on which PLCC and RMSE are then reported.
    """
    pred_values = _to_finite_float_array(pred)
    target_values = _to_finite_float_array(target)
    if len(pred_values) != len(target_values):
        raise ValueError("Prediction and target lengths must match")
    if len(pred_values) < 6 or np.std(pred_values) == 0 or np.std(target_values) == 0:
        raise ValueError("Five-parameter mapping requires at least six non-constant samples")

    target_range = float(np.ptp(target_values))
    direction = 1.0 if _safe_correlation(spearmanr, pred_values, target_values) >= 0 else -1.0
    initial = (
        target_range or 1.0,
        direction / max(float(np.std(pred_values)), 1e-6),
        float(np.median(pred_values)),
        0.0,
        float(np.mean(target_values)),
    )
    if direction > 0:
        lower = (0.0, 0.0, -np.inf, 0.0, -np.inf)
        upper = (np.inf, np.inf, np.inf, np.inf, np.inf)
    else:
        lower = (0.0, -np.inf, -np.inf, -np.inf, -np.inf)
        upper = (np.inf, 0.0, np.inf, 0.0, np.inf)
    parameters, _ = curve_fit(
        five_parameter_logistic,
        pred_values,
        target_values,
        p0=initial,
        bounds=(lower, upper),
        maxfev=100_000,
    )
    mapped = five_parameter_logistic(pred_values, *parameters)
    beta1, beta2, beta3, beta4, beta5 = (float(value) for value in parameters)
    return mapped, (beta1, beta2, beta3, beta4, beta5)


def compute_mapped_metrics(
    pred: torch.Tensor | np.ndarray | list[float],
    target: torch.Tensor | np.ndarray | list[float],
) -> MappedMetrics:
    """Report raw metrics plus PLCC/RMSE/MAE after five-parameter mapping."""
    raw = compute_metrics(pred, target)
    try:
        mapped_values, parameters = fit_five_parameter_logistic(pred, target)
        mapped = compute_metrics(mapped_values, target)
        return {
            "raw": raw,
            "mapped": {key: mapped[key] for key in ("plcc", "rmse", "mae")},
            "mapping": {
                "applied": True,
                "method": "five_parameter_logistic",
                "parameters": list(parameters),
            },
        }
    except (RuntimeError, TypeError, ValueError, FloatingPointError) as exc:
        return {
            "raw": raw,
            "mapped": None,
            "mapping": {
                "applied": False,
                "method": "five_parameter_logistic",
                "error": str(exc),
            },
        }


__all__ = [
    "compute_mapped_metrics",
    "compute_metrics",
    "fit_five_parameter_logistic",
    "five_parameter_logistic",
]
