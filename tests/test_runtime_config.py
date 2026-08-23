from deploy.core.runtime_config import InferenceConfig, ServiceEndpoints


def test_service_endpoints_use_shared_environment_defaults(monkeypatch):
    monkeypatch.setenv("API_PORT", "9101")
    monkeypatch.setenv("WEB_PORT", "9100")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://vqa-ollama:11434")
    monkeypatch.setenv("DEEP_VQA_SQLITE_PATH", "tmp/evaluations.sqlite3")

    endpoints = ServiceEndpoints()

    assert endpoints.api_port == 9101
    assert endpoints.web_port == 9100
    assert endpoints.ollama_base_url == "http://vqa-ollama:11434"
    assert str(endpoints.sqlite_path) == "tmp/evaluations.sqlite3"


def test_inference_config_resolves_registry_paths(tmp_path):
    config = InferenceConfig(project_root=tmp_path, endpoints=ServiceEndpoints(sqlite_path="reports/test.sqlite3"))

    assert config.resolve(config.endpoints.sqlite_path) == tmp_path / "reports/test.sqlite3"


def test_profile_selection_is_required(monkeypatch):
    monkeypatch.delenv("DEEP_VQA_DEPLOYMENT_CONFIG", raising=False)
    from deploy.core.runtime_config import _load_deployment_policy

    try:
        _load_deployment_policy()
    except RuntimeError as exc:
        assert "DEEP_VQA_DEPLOYMENT_CONFIG" in str(exc)
    else:
        raise AssertionError("deployment profile selection must be explicit")
