"""Shared checkpoint reconstruction retains safe loading and serving semantics."""

import subprocess
import sys

import torch

from src.config.loader import load_config
from src.utils.checkpointing import load_checkpoint


def test_checkpoint_reconstructs_config_weights_and_eval_mode(tmp_path, monkeypatch):
    from src.models import iqavqa_model

    config = load_config()
    original = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config.model_dump(mode="json"), "state_dict": original.state_dict()}, checkpoint)
    monkeypatch.setattr(iqavqa_model, "IQAVQANet", lambda **_kwargs: torch.nn.Linear(2, 1))
    real_load = torch.load
    calls = []

    def safe_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    model, restored = load_checkpoint(checkpoint, "cpu")
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert restored.model_dump() == config.model_dump()
    assert not model.training
    torch.testing.assert_close(model.weight, original.weight)
    torch.testing.assert_close(model.bias, original.bias)


def test_offline_eda_import_has_no_deployment_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.data.eda.feature_distribution; assert not any(name == 'deploy' or name.startswith('deploy.') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_training_serialization_keeps_string_enums_safe_for_weights_only_loading(tmp_path):
    from src.core.engine import TrainerEngine

    config = load_config()
    engine = TrainerEngine.__new__(TrainerEngine)
    checkpoint = tmp_path / "training.pt"
    serialized = engine._sanitize_for_serialization(config)
    torch.save({"config": serialized}, checkpoint)
    restored = torch.load(checkpoint, weights_only=True)["config"]
    assert type(restored["task_type"]) is str
    assert type(restored["dataset"]["data_type"]) is str
    assert restored == config.model_dump(mode="json")
