"""Closed choices and public history remain consistent with the HTTP contract."""

import json

import pytest
from pydantic import ValidationError

from deploy.config import AuthSettings, EvaluationStoreSettings, ServingModelSettings
from deploy.types import AuthMode
from deploy.web.schemas import AuthStatusResponse, FrontendEvaluationLogResponse


def test_auth_choices_validate_and_keep_json_strings():
    for mode in AuthMode:
        settings = AuthSettings(mode=mode.value)
        assert settings.mode is mode
        response = AuthStatusResponse(required=True, authenticated=False, mode=settings.mode)
        assert json.loads(response.model_dump_json())["mode"] == mode.value
    with pytest.raises(ValidationError):
        AuthSettings(mode="accoutns")
    with pytest.raises(ValidationError):
        AuthStatusResponse(required=True, authenticated=False, mode="accoutns")


def test_model_and_backend_choices_reject_unknown_values():
    fields = dict(
        checkpoint_path="model.pt",
        task_type="iqa",
        media_type="image",
        model_name="test",
        backbone="test",
        dataset="test",
        mos_min=0,
        mos_max=5,
    )
    for field, value in (("task_type", "unknown"), ("media_type", "audio")):
        with pytest.raises(ValidationError):
            ServingModelSettings(**(fields | {field: value}))
    with pytest.raises(ValidationError):
        EvaluationStoreSettings(backend="unknown")


def test_history_has_typed_items_and_two_number_json_interval():
    record = dict(
        timestamp="2026-10-29",
        file_name="sample.jpg",
        file_hash="a" * 64,
        task_type="iqa",
        model_used="test",
        mos_score=None,
        mos_interval=[0, 5],
        inference_time_ms=1,
    )
    response = FrontendEvaluationLogResponse(items=[record], count=1)
    payload = json.loads(response.model_dump_json())
    assert payload["items"][0] == record
    assert "session_id" not in payload["items"][0]
    for changes in ({"task_type": "unknown"}, {"mos_interval": [0, 1, 5]}, {"inference_time_ms": -1}):
        with pytest.raises(ValidationError):
            FrontendEvaluationLogResponse(items=[record | changes], count=1)
