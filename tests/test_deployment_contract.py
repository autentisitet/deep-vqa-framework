from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compose_port_contract_is_explicit() -> None:
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")
    assert '"${API_BIND_HOST:-127.0.0.1}:${API_PORT:-8001}:${API_INTERNAL_PORT:-8000}"' in compose
    assert '"${WEB_BIND_HOST:-127.0.0.1}:${WEB_PORT:-8000}:${WEB_INTERNAL_PORT:-80}"' in compose


def test_ollama_compose_has_service_and_health_contract() -> None:
    compose = (ROOT / "deploy-config/ollama/docker-compose.yaml").read_text(encoding="utf-8")
    assert "vqa-ollama:" in compose
    assert "condition: service_healthy" not in compose
    assert "ollama list" in compose


def test_make_stop_and_purge_include_ollama_without_stop_alias() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "docker-ollama-stop" not in makefile
    assert "stop vqa-ollama" in makefile
    assert "OLLAMA_COMPOSE_FILES) down --remove-orphans --volumes" in makefile


def test_profile_targets_select_matching_policy_files() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    internal = (ROOT / "deploy-config/profiles/infer_deploy.internal.yaml").read_text(encoding="utf-8")
    public = (ROOT / "deploy-config/profiles/infer_deploy.public.yaml").read_text(encoding="utf-8")

    assert "docker-infer-internal:" in makefile
    assert "run_infer_profile,deploy-config/profiles/infer_deploy.internal.yaml" in makefile
    assert "docker-infer-public:" in makefile
    assert "run_infer_profile,deploy-config/profiles/infer_deploy.public.yaml" in makefile
    assert "mode: api_key" in internal
    assert "backend: sqlite" in internal
    assert "mode: none" in public
    assert "backend: none" in public


def test_prod_image_copies_all_deployment_profiles() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY deploy-config/ ./deploy-config/" in dockerfile


def test_default_deployment_profile_was_removed() -> None:
    assert not (ROOT / "deploy-config/profiles/infer_deploy.yaml").exists()
    runtime = (ROOT / "deploy/core/runtime_config.py").read_text(encoding="utf-8")
    assert 'os.getenv("DEEP_VQA_DEPLOYMENT_CONFIG", "").strip()' in runtime


def test_deployment_check_uses_state_aware_service_budgets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert '[ -n "$$health" ] || health=unconfigured' in makefile
    assert 'check_http vqa-infer "$(API_HEALTH_URL)" no 60' in makefile
    assert 'check_http vqa-nginx "$(WEB_HEALTH_URL)" yes 30' in makefile
    assert 'check_http vqa-ollama "$(OLLAMA_HEALTH_URL)" yes 60' in makefile


def test_make_pytest_requires_dev_tool_without_resync() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "$(call require_tool,pytest,dev)" in makefile
    assert "uv run --no-sync pytest -q" in makefile
