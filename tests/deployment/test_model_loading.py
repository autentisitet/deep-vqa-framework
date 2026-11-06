from types import SimpleNamespace

import pytest

api = pytest.importorskip("deploy.web.api")


def _fake_config(model_id: str):
    settings = api.MODEL_SETTINGS[model_id]
    return SimpleNamespace(
        task_type=settings.task_type,
        dataset=SimpleNamespace(
            name=settings.dataset,
            registry_key=settings.dataset,
            mos_min=settings.mos_min,
            mos_max=settings.mos_max,
        ),
        model=SimpleNamespace(name=settings.model_name, backbone=settings.backbone, input_size=224),
    )


def test_worker_preloads_each_model_once_and_requests_only_reuse(monkeypatch, tmp_path) -> None:
    checkpoints = {model_id: tmp_path / f"{model_id}.pt" for model_id in api.MODEL_SETTINGS}
    for path in checkpoints.values():
        path.write_bytes(b"checkpoint")
    loaded = []

    def fake_load_checkpoint(path, device):
        model_id = path.stem
        loaded.append((model_id, device))
        return object(), _fake_config(model_id)

    monkeypatch.setattr(api, "MODEL_PATHS", checkpoints)
    monkeypatch.setattr(api, "load_checkpoint", fake_load_checkpoint)
    api.model_cache.clear()

    api.load_all_models()
    first_iqa = api._loaded_model("iqa")
    second_iqa = api._loaded_model("iqa")
    api._loaded_model("vqa")

    assert first_iqa is second_iqa
    assert set(api.model_cache) == {"iqa", "vqa"}
    assert loaded == [("iqa", api.DEVICE), ("vqa", api.DEVICE)]


def test_model_list_reports_loaded_models(monkeypatch) -> None:
    monkeypatch.setattr(api, "model_cache", {model_id: {} for model_id in api.MODEL_SETTINGS})

    response = __import__("asyncio").run(api.list_models())

    assert response.count == 2
    assert {item.model_id for item in response.items} == {"iqa", "vqa"}
    assert all(item.loaded is True for item in response.items)


def test_public_model_summary_does_not_expose_server_checkpoint_directory(monkeypatch, tmp_path) -> None:
    model_file = tmp_path / "private" / "tid2013_best.pt"
    monkeypatch.setattr(api.MODEL_SETTINGS["iqa"], "checkpoint_path", model_file)
    summary = api._model_summary("iqa")
    assert summary.checkpoint == "tid2013_best.pt"
    assert str(tmp_path) not in summary.model_dump_json()


def test_validate_model_catalog_rejects_missing_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "MODEL_PATHS", {"iqa": tmp_path / "missing.pt"})

    with pytest.raises(RuntimeError, match="checkpoints are missing"):
        api.validate_model_catalog()


def test_model_load_rejects_any_policy_metadata_drift(monkeypatch, tmp_path) -> None:
    config = _fake_config("iqa")
    config.model.backbone = "resnet50"
    monkeypatch.setattr(api, "load_checkpoint", lambda *_args, **_kwargs: (object(), config))

    with pytest.raises(ValueError, match="backbone"):
        api._load_model("iqa", tmp_path / "iqa.pt")


def test_heavy_operation_respects_serialization_policy(monkeypatch) -> None:
    class FakeLock:
        def __init__(self):
            self.entries = 0

        async def __aenter__(self):
            self.entries += 1

        async def __aexit__(self, *_args):
            return None

    async def run_once():
        async with api._heavy_operation():
            pass

    lock = FakeLock()
    monkeypatch.setattr(api, "_heavy_task_lock", lock)
    monkeypatch.setattr(api.cfg.deployment.inference_runtime, "serialize_heavy_requests", True)
    __import__("asyncio").run(run_once())
    assert lock.entries == 1

    monkeypatch.setattr(api.cfg.deployment.inference_runtime, "serialize_heavy_requests", False)
    __import__("asyncio").run(run_once())
    assert lock.entries == 1
