"""Decorators preserve calls, cleanup, and computed properties remain live."""

import inspect

import numpy as np
import pytest

from src.config.loader import load_config
from src.core.evaluator import Evaluator
from src.core.trainer import release_torch_cache
from src.utils.logging_utils import time_it


def test_training_decorators_preserve_signature_results_and_cleanup_on_error(monkeypatch):
    from src.core import trainer

    cleanup = []
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "empty_cache", lambda: cleanup.append("cuda"))
    monkeypatch.setattr(trainer.gc, "collect", lambda: cleanup.append("gc"))

    def stage(value: int, *, fail: bool = False) -> int:
        if fail:
            raise ValueError("stage failed")
        return value + 1

    wrapped = release_torch_cache(time_it(stage))
    assert inspect.signature(wrapped) == inspect.signature(stage)
    assert wrapped(3) == 4
    assert cleanup == ["cuda", "gc"]
    with pytest.raises(ValueError, match="stage failed"):
        wrapped(3, fail=True)
    assert cleanup == ["cuda", "gc", "cuda", "gc"]


def test_evaluator_properties_follow_path_and_filename_changes(tmp_path):
    evaluator = Evaluator(load_config(), tmp_path)
    evaluator.base_filename = "first"
    assert evaluator.history_path == tmp_path / "first_history.csv"
    evaluator.history_path = tmp_path / "next" / "second_history.csv"
    assert evaluator.base_filename == "second"
    assert evaluator.manifest_path == tmp_path / "next" / "second_manifest.csv"
    evaluator.manifest_path = tmp_path / "third_manifest.csv"
    assert evaluator.history_path == tmp_path / "third_history.csv"


def test_object_metric_input_rejects_unknown_objects_without_array_coercion():
    evaluator = Evaluator.__new__(Evaluator)
    np.testing.assert_array_equal(evaluator._to_clean_numpy([1, 2]), [1, 2])
    with pytest.raises(TypeError, match="Unsupported metric input"):
        evaluator._to_clean_numpy(object())


def test_deployment_decorator_checks_unknown_context_before_using_it(monkeypatch):
    from src import main

    deployed = []
    monkeypatch.setattr(main, "_deploy_best_checkpoint", lambda **context: deployed.append(context))

    @main.deploy_best_checkpoint_after_run
    def stage(context: object) -> object:
        return context

    for context in (False, None, {"cfg": object(), "dataset_name": "test", "base_fn": "run"}):
        assert stage(context) is context
    assert not deployed
    config = load_config()
    valid = {"cfg": config, "dataset_name": "tid2013", "base_fn": "run"}
    assert stage(valid) is valid
    assert deployed == [valid]
