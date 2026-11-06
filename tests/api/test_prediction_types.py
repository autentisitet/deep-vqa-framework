"""Inference validates unknown objects before using their config fields."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from deploy.inference.predictor import _ensure_dict, predict_single


class ConstantQuality(torch.nn.Module):
    def forward(self, image):
        return torch.full((image.shape[0], 1), 0.5)


def test_prediction_accepts_validated_mapping_and_preserves_output_shape(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), (80, 80, 80)).save(image)
    result = predict_single(
        ConstantQuality(),
        image,
        {
            "model": {"input_size": 8, "num_frames": 2, "name": "constant"},
            "dataset": {"mos_min": 1, "mos_max": 5},
            "task_type": "iqa",
        },
        device="cpu",
    )
    assert result == {
        "file": str(image),
        "raw_score": 0.5,
        "mos_score": 3.0,
        "task_type": "iqa",
        "model_name": "constant",
    }


@pytest.mark.parametrize("value", [object(), {1: "invalid key"}, [], "invalid"])
def test_unknown_config_objects_are_rejected(value):
    with pytest.raises(TypeError, match="string-keyed configuration"):
        _ensure_dict(value)


@pytest.mark.parametrize("value", [True, "8", 0, -1])
def test_bad_preprocessing_values_fail_before_media_io(value):
    with pytest.raises(TypeError, match="input_size must be a positive integer"):
        predict_single(ConstantQuality(), Path("missing.png"), {"model": {"input_size": value}}, device="cpu")


@pytest.mark.parametrize("value", [[], 0, False, "invalid"])
def test_non_mapping_nested_config_is_rejected(value):
    with pytest.raises(TypeError, match="string-keyed configuration"):
        predict_single(ConstantQuality(), Path("missing.png"), {"model": value}, device="cpu")
