from pathlib import Path

import pytest

from deploy.config import (
    InferenceConfig,
    ServiceEndpoints,
    _ensure_writable_directory,
    ensure_runtime_dirs,
)


def test_service_endpoints_use_shared_environment_defaults(monkeypatch):
    monkeypatch.setenv("API_PORT", "9101")
    monkeypatch.setenv("WEB_PORT", "9100")
    monkeypatch.setenv("DEEP_VQA_SQLITE_PATH", "tmp/evaluations.sqlite3")
    monkeypatch.setenv("DEEP_VQA_ACCOUNTS_SQLITE_PATH", "tmp/accounts.sqlite3")

    endpoints = ServiceEndpoints()

    assert endpoints.api_port == 9101
    assert endpoints.web_port == 9100
    assert str(endpoints.sqlite_path) == "tmp/evaluations.sqlite3"
    assert str(endpoints.accounts_sqlite_path) == "tmp/accounts.sqlite3"


def test_upload_directory_defaults_to_project_relative_path(monkeypatch):
    monkeypatch.delenv("DEEP_VQA_UPLOAD_DIR", raising=False)

    assert ServiceEndpoints().upload_dir == Path("uploads")


def test_inference_config_resolves_registry_paths(tmp_path):
    config = InferenceConfig(project_root=tmp_path, endpoints=ServiceEndpoints(sqlite_path="reports/test.sqlite3"))

    assert config.resolve_path(config.endpoints.sqlite_path) == tmp_path / "reports/test.sqlite3"


def test_runtime_directory_setup_creates_and_probes_upload_directory(tmp_path):
    config = InferenceConfig(
        project_root=tmp_path,
        endpoints=ServiceEndpoints(sqlite_path="reports/test.sqlite3", upload_dir="runtime/uploads"),
    )

    ensure_runtime_dirs(config)

    upload_dir = tmp_path / "runtime/uploads"
    assert upload_dir.is_dir()
    assert list(upload_dir.iterdir()) == []


def test_writable_directory_probe_reports_startup_error(monkeypatch, tmp_path):
    def fail_probe(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("deploy.config.tempfile.mkstemp", fail_probe)

    with pytest.raises(RuntimeError, match="Runtime directory is not writable"):
        _ensure_writable_directory(tmp_path / "uploads")


def test_profile_selection_is_required(monkeypatch):
    monkeypatch.delenv("DEEP_VQA_DEPLOYMENT_CONFIG", raising=False)
    from deploy.config import _load_deployment_settings

    try:
        _load_deployment_settings()
    except RuntimeError as exc:
        assert "DEEP_VQA_DEPLOYMENT_CONFIG" in str(exc)
    else:
        raise AssertionError("deployment mode configuration must be explicit")


@pytest.mark.parametrize(
    "mode,auth,backend", [("internal", "accounts", "sqlite"), ("local", "none", "sqlite"), ("proxy", "none", "none")]
)
def test_mode_profiles_inherit_shared_defaults(mode, auth, backend):
    from deploy.config import DeploymentSettings, load_deployment_profile

    root = Path(__file__).resolve().parents[2]
    policy = DeploymentSettings.model_validate(
        load_deployment_profile(root / f"deploy-config/profiles/infer_deploy.{mode}.yaml")
    )
    assert policy.auth.mode == auth
    assert policy.auth.login_attempts == 5
    assert policy.evaluation_store.backend == backend
    assert policy.uploads.max_bytes == 104857600
    assert policy.sqlite.max_database_bytes == 268435456
    assert policy.artifacts.retention_seconds == 604800
    assert set(policy.models) == {"iqa", "vqa"}
    assert policy.models["iqa"].mos_min == 0.242
    assert policy.models["vqa"].mos_max == 4.64


def test_profile_nested_override_and_schema_validation(tmp_path):
    from pydantic import ValidationError

    from deploy.config import DeploymentSettings, load_deployment_profile

    root = Path(__file__).resolve().parents[2]
    (tmp_path / "basic.yaml").write_text((root / "deploy-config/basic.yaml").read_text())
    profile = tmp_path / "profiles/custom.yaml"
    profile.parent.mkdir()
    profile.write_text("auth:\n  mode: accounts\n  login_attempts: 8\nuploads:\n  max_bytes: 1024\n")
    settings = load_deployment_profile(profile)
    policy = DeploymentSettings.model_validate(settings)
    assert policy.uploads.max_bytes == 1024
    assert policy.uploads.read_timeout_seconds == 30
    assert policy.auth.login_attempts == 8
    assert policy.auth.login_window_seconds == 900
    profile.write_text("auth: {mode: accounts}\nuploads:\n  typo: 1\n")
    with pytest.raises(ValidationError, match="typo"):
        DeploymentSettings.model_validate(load_deployment_profile(profile))


@pytest.mark.parametrize("content", ["[]", "false", "null", "", "auth: [", "{}", "auth: {}"])
def test_invalid_profile_is_rejected(tmp_path, content):
    from deploy.config import load_deployment_profile

    (tmp_path / "basic.yaml").write_text("auth: {}")
    profile = tmp_path / "profiles/custom.yaml"
    profile.parent.mkdir()
    profile.write_text(content)
    with pytest.raises(RuntimeError, match="configuration|Configuration"):
        load_deployment_profile(profile)


def test_missing_shared_defaults_is_an_error(tmp_path):
    from deploy.config import load_deployment_profile

    profile = tmp_path / "profiles/custom.yaml"
    profile.parent.mkdir()
    profile.write_text("auth: {mode: accounts}")
    with pytest.raises(RuntimeError, match="basic.yaml"):
        load_deployment_profile(profile)
