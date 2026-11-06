import numpy as np
import pytest
import torch

from src.models.metrics import (
    compute_mapped_metrics,
    compute_metrics,
    fit_five_parameter_logistic,
    five_parameter_logistic,
)


def test_compute_metrics_matches_known_values_and_reports_krocc() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    prediction = torch.tensor([1.0, 2.0, 4.0, 3.0])

    metrics = compute_metrics(prediction, target)

    assert metrics["plcc"] == pytest.approx(0.8)
    assert metrics["srocc"] == pytest.approx(0.8)
    assert metrics["krocc"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mae"] == pytest.approx(0.5)


def test_compute_metrics_handles_constant_and_invalid_inputs() -> None:
    assert compute_metrics([1, 1, 1], [2, 3, 4])["plcc"] == 0.0
    with pytest.raises(ValueError, match="lengths must match"):
        compute_metrics([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        compute_metrics([1, np.nan], [1, 2])


def test_five_parameter_mapping_recovers_nonlinear_mos_relation() -> None:
    scores = np.linspace(-2.0, 2.0, 40)
    target = five_parameter_logistic(scores, 4.2, 1.6, 0.1, 0.25, 3.0)
    raw = compute_metrics(scores, target)

    mapped, parameters = fit_five_parameter_logistic(scores, target)
    report = compute_mapped_metrics(scores, target)

    assert len(parameters) == 5
    assert compute_metrics(mapped, target)["plcc"] > 0.99999
    assert report["mapping"]["applied"] is True
    assert report["mapped"]["plcc"] > raw["plcc"]
    assert report["mapped"]["rmse"] < raw["rmse"]
    assert report["raw"]["srocc"] == pytest.approx(1.0)
    assert report["raw"]["krocc"] == pytest.approx(1.0)


def test_five_parameter_mapping_remains_monotonic() -> None:
    scores = np.linspace(0.0, 10.0, 30)
    target = np.tanh(scores - 5.0) + 0.05 * scores

    mapped, _ = fit_five_parameter_logistic(scores, target)

    assert np.all(np.diff(mapped) >= -1e-8)


def test_mapping_reports_fallback_for_too_few_samples() -> None:
    report = compute_mapped_metrics([1, 2, 3], [1, 2, 3])

    assert report["mapping"]["applied"] is False
    assert report["mapped"] is None
