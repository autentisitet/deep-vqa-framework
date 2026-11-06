import os
from pathlib import Path

# tests/deployment/test_*.py -> repository root
ROOT = Path(__file__).resolve().parents[2]


def _repository_files(name: str) -> list[Path]:
    """Find a filename without walking dependency, cache, or build directories."""
    skipped = {".git", ".venv", ".cache", "node_modules", "__pycache__", "data", "datasets"}
    found: list[Path] = []
    for directory, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [item for item in dirnames if item not in skipped]
        if name in filenames:
            found.append(Path(directory) / name)
    return found


def test_all_compose_services_use_bounded_json_log_rotation() -> None:
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")

    # vqa-dev, vqa-train, vqa-infer, vqa-nginx
    assert compose.count("    logging:") == 4
    assert compose.count('driver: "json-file"') == 4
    assert compose.count('max-size: "${CONTAINER_LOG_MAX_SIZE:-50m}"') == 4
    assert compose.count('max-file: "${CONTAINER_LOG_MAX_FILES:-3}"') == 4


def test_auth_rate_limits_and_private_nginx_config_are_present() -> None:
    nginx = (ROOT / "deploy-config/nginx/nginx.conf").read_text(encoding="utf-8")
    default = (ROOT / "deploy-config/nginx/default.conf").read_text(encoding="utf-8")
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")

    assert "limit_req_zone $binary_remote_addr zone=auth_login" in nginx
    assert "limit_req_zone $binary_remote_addr zone=api_requests" in nginx
    assert "limit_req zone=auth_login" in default
    assert "limit_req zone=api_requests" in default
    assert "nginx.conf:/etc/nginx/nginx.conf:ro" in compose
    from deploy.config import load_deployment_profile

    policy = load_deployment_profile(ROOT / "deploy-config/profiles/infer_deploy.internal.yaml")
    assert policy["auth"]["login_attempts"] == 5


def test_missing_static_paths_and_bare_api_prefix_do_not_redirect_or_serve_homepage() -> None:
    default = (ROOT / "deploy-config/nginx/default.conf").read_text(encoding="utf-8")
    assert "location = /api {" in default
    assert "return 404;" in default
    assert "try_files $uri $uri/ =404;" in default
    assert "try_files $uri $uri/ /index.html;" not in default


def test_nginx_bounds_request_line_headers_and_reports_throttling_as_429() -> None:
    nginx = (ROOT / "deploy-config/nginx/nginx.conf").read_text(encoding="utf-8")
    assert "client_header_buffer_size 1k;" in nginx
    assert "large_client_header_buffers 4 8k;" in nginx
    assert "limit_req_status 429;" in nginx
    assert "limit_conn_status 429;" in nginx


def test_nginx_upload_limit_aligns_with_application_limit() -> None:
    nginx = (ROOT / "deploy-config/nginx/default.conf").read_text(encoding="utf-8")

    assert "client_max_body_size 101m;" in nginx
    assert "client_body_timeout 30s;" in nginx
    assert "proxy_request_buffering off;" in nginx


def test_web_frontend_and_api_use_one_origin_without_cors() -> None:
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")
    api = (ROOT / "deploy/web/api.py").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "CORS_ALLOW_ORIGINS" not in compose + api + env_example
    assert "CORSMiddleware" not in api
    for page in ("index.html", "login.html"):
        assert "deep-vqa-api-base" not in (ROOT / "frontend" / page).read_text(encoding="utf-8")
    for script in ("app.js", "login.js"):
        content = (ROOT / "frontend" / script).read_text(encoding="utf-8")
        assert 'const API = "/api/v1";' in content
        assert "API_STORAGE_KEY" not in content


def test_local_tls_front_proxy_preserves_browser_scheme() -> None:
    nginx = (ROOT / "deploy-config/nginx/nginx.conf").read_text(encoding="utf-8")
    default = (ROOT / "deploy-config/nginx/default.conf").read_text(encoding="utf-8")
    assert "map $http_x_forwarded_proto $client_proto" in nginx
    assert "https https;" in nginx
    assert default.count("proxy_set_header X-Forwarded-Proto $client_proto;") == 2
    assert "proxy_set_header X-Forwarded-Proto $scheme;" not in default


def test_nginx_forwards_the_client_host_including_port() -> None:
    """The application's same-origin check compares Origin against Host.

    nginx's ``$host`` drops the port, so a browser reaching the proxy on a
    non-default port sends ``Origin: http://host:port`` while the application
    reconstructs ``http://host`` and rejects every authenticated state-changing
    request with 403. ``$http_host`` forwards the client's Host verbatim.
    """
    default = (ROOT / "deploy-config/nginx/default.conf").read_text(encoding="utf-8")

    assert "proxy_set_header Host $http_host;" in default
    assert "proxy_set_header Host $host;" not in default


def test_compose_port_contract_is_explicit() -> None:
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")
    assert '"${WEB_BIND_HOST:-127.0.0.1}:${WEB_PORT:-8000}:${WEB_INTERNAL_PORT:-80}"' in compose
    assert "vqa-infer:" in compose
    assert "expose:" in compose
    assert "vqa-uploads:/app/uploads" in compose
    # nginx serves only frontend/; repository documentation is linked out to the
    # source host instead of being mounted under the web root.
    assert "../../frontend:/usr/share/nginx/html:ro,Z" in compose
    assert "${IQA_MODEL_VOLUME_PATH:-/opt/deep-vqa/deploy/iqa-models}:/app/deploy/iqa-models:ro,z" in compose
    assert "${VQA_MODEL_VOLUME_PATH:-/opt/deep-vqa/deploy/vqa-models}:/app/deploy/vqa-models:ro,z" in compose
    assert 'test: ["CMD", "nginx", "-t"]' in compose


def test_api_explorer_is_loopback_only_and_not_promoted_in_frontend() -> None:
    nginx = (ROOT / "deploy-config/nginx/default.conf").read_text(encoding="utf-8")
    index = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")
    api = (ROOT / "deploy/web/api.py").read_text(encoding="utf-8")
    for path in ("/api/docs", "/api/redoc", "/api/openapi.json"):
        assert path in nginx
        assert path not in index
    assert "${API_BIND_HOST:-127.0.0.1}:${API_PORT:-8001}" in compose
    assert 'docs_url="/docs"' in api
    assert 'redoc_url="/redoc"' in api


def test_production_image_copies_all_required_deploy_modules() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY deploy/types.py ./deploy/types.py" in dockerfile
    assert "COPY deploy/config.py ./deploy/config.py" in dockerfile
    assert "COPY deploy/web/ ./deploy/web/" in dockerfile
    assert "COPY deploy/inference/ ./deploy/inference/" in dockerfile


def test_inference_mounts_only_runtime_inputs_and_persistent_outputs() -> None:
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")
    inference = compose.split("  vqa-infer:", 1)[1].split("  vqa-nginx:", 1)[0]
    assert "../../results:/app/results" not in inference
    assert "extra_hosts:" not in inference
    assert "host-gateway" not in inference
    assert "vqa-uploads:/app/uploads" in inference
    assert "../../reports:/app/reports:z" in inference
    assert "../../deploy-config:/app/deploy-config:ro,z" in inference
    assert "../../.cache:/app/.cache:z" in inference


def test_training_container_can_publish_checkpoints() -> None:
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")
    training = compose.split("  vqa-train:", 1)[1].split("  vqa-infer:", 1)[0]
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    train_target = makefile.split("docker-train:", 1)[1].split("\n\n", 1)[0]

    assert "../../deploy/iqa-models:/app/deploy/iqa-models" in training
    assert "../../deploy/vqa-models:/app/deploy/vqa-models" in training
    assert ":ro" not in training
    assert "../../docs:/app/docs" not in training
    assert "$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES)" in train_target


def test_base_topology_owns_cross_engine_selinux_mount_labels() -> None:
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")

    for mount in (
        "../../reports:/app/reports:z",
        "../../.cache:/app/.cache:z",
        "../../docs:/app/docs:z",
        "../../deploy/iqa-models:/app/deploy/iqa-models:z",
        "../../deploy/vqa-models:/app/deploy/vqa-models:z",
        "../../deploy-config:/app/deploy-config:ro,z",
        "../../frontend:/usr/share/nginx/html:ro,Z",
    ):
        assert mount in compose


def test_cpu_is_default_and_gpu_requires_explicit_opt_in() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "GPU ?= 0" in makefile
    assert "GPU=1                      Enable NVIDIA GPU access" in makefile
    assert "GPU_DISABLED := $(filter 0 false no,$(strip $(GPU)))" in makefile


def test_cpu_overlays_are_empty_engine_accelerator_differences() -> None:
    for engine in ("docker", "podman"):
        overlay = (ROOT / f"deploy-config/compose/docker-compose.{engine}-cpu.yaml").read_text(encoding="utf-8")
        assert "services: {}" in overlay
        assert "volumes:" not in overlay
        assert "runtime:" not in overlay


def test_language_explanation_service_is_fully_removed() -> None:
    """The optional VLM explanation service is absent from every code path."""
    surfaces = {
        "compose": ROOT / "deploy-config/compose/docker-compose.yaml",
        "makefile": ROOT / "Makefile",
        "api": ROOT / "deploy/web/api.py",
        "runtime_config": ROOT / "deploy/config.py",
        "internal_profile": ROOT / "deploy-config/profiles/infer_deploy.internal.yaml",
        "proxy_profile": ROOT / "deploy-config/profiles/infer_deploy.proxy.yaml",
        "env_example": ROOT / ".env.example",
    }
    for name, path in surfaces.items():
        content = path.read_text(encoding="utf-8").lower()
        assert "vqa-language" not in content, f"{name} still references vqa-language"
        assert "llama" not in content, f"{name} still references llama"

    assert not (ROOT / "deploy-config/llama-cpp").exists()
    assert not (ROOT / "docs/llama_cpp.md").exists()


def test_make_lifecycle_covers_the_remaining_stack() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "OLLAMA=1" not in makefile
    assert "host-infer:" not in makefile
    assert "stop vqa-infer" in makefile


def test_docker_manage_is_read_only() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    manage = makefile.split("\ndocker-manage:\n", 1)[1].split("\ndocker-purge-train:\n", 1)[0]
    assert "rmi" not in manage
    assert "prune" not in manage
    assert "rm -f" not in manage


def test_project_purge_does_not_prune_unrelated_runtime_resources() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    purge = makefile.split("\ndocker-purge-all:\n", 1)[1]
    assert "$(COMPOSE) $(COMPOSE_FILES) down --volumes" in purge
    assert "rmi vqa-train:latest vqa-dev:latest vqa-infer:latest" in purge
    assert "--remove-orphans" not in purge
    assert "volume prune" not in purge
    assert "network prune" not in purge
    assert "dangling=true" not in purge
    assert "rm -rf" not in purge


def test_scoped_purge_targets_never_prune_other_projects_or_bind_mounted_data() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    train = makefile.split("\ndocker-purge-train:\n", 1)[1].split("\ndocker-purge-deploy:\n", 1)[0]
    deploy = makefile.split("\ndocker-purge-deploy:\n", 1)[1].split("\ndocker-purge-all:\n", 1)[0]

    for target in (train, deploy):
        assert "$(call check_runtime)" in target
        assert "volume prune" not in target
        assert "network prune" not in target
        assert "--volumes" not in target
        assert "--remove-orphans" not in target
        assert "rm -rf" not in target
        assert "rmi -f" not in target

    assert "rm -f vqa-train vqa-dev" in train
    assert "rmi vqa-train:latest vqa-dev:latest" in train
    assert "vqa-infer" not in train and "vqa-nginx" not in train
    assert "rm -f vqa-nginx vqa-infer" in deploy
    assert "rmi vqa-infer:latest" in deploy
    assert "vqa-train" not in deploy and "vqa-dev" not in deploy


def test_local_profile_runs_without_login_and_keeps_history() -> None:
    local = (ROOT / "deploy-config/profiles/infer_deploy.local.yaml").read_text(encoding="utf-8")

    assert "mode: none" in local
    assert "backend: sqlite" in local

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\nlocal:\n" in makefile
    assert "DEPLOYMENT_MODE=local" in makefile
    assert "stop vqa-nginx" in makefile
    assert "docker-batch-test:" in makefile
    assert "docker-memory-measure:" in makefile
    assert "scripts/measure_container_memory.py" in makefile


def test_make_replaces_nginx_before_its_api_dependency() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    nginx_remove = makefile.index("rm -f vqa-nginx")
    infer_remove = makefile.index("rm -f vqa-infer", nginx_remove)

    assert nginx_remove < infer_remove


def test_nginx_start_preserves_selected_deployment_profile() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) up -d vqa-nginx" in makefile


def test_modes_select_matching_configuration_files() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    internal = (ROOT / "deploy-config/profiles/infer_deploy.internal.yaml").read_text(encoding="utf-8")
    proxy = (ROOT / "deploy-config/profiles/infer_deploy.proxy.yaml").read_text(encoding="utf-8")

    assert "run_infer_mode,$(DEPLOYMENT_CONFIG)" in makefile
    assert "docker-infer-internal:" not in makefile
    assert "docker-infer-public:" not in makefile
    assert "DEPLOYMENT_MODE must be internal, proxy, or local" in makefile
    assert "mode: accounts" in internal
    assert "backend: sqlite" in internal
    from deploy.config import load_deployment_profile

    effective = load_deployment_profile(ROOT / "deploy-config/profiles/infer_deploy.internal.yaml")
    assert effective["inference_runtime"]["preload_models"] is True
    assert effective["inference_runtime"]["serialize_heavy_requests"] is True
    assert "mode: none" in proxy
    assert "backend: none" in proxy
    local = (ROOT / "deploy-config/profiles/infer_deploy.local.yaml").read_text(encoding="utf-8")
    assert "mode: none" in local
    assert "backend: sqlite" in local
    assert not (ROOT / "deploy-config/profiles/infer_deploy.edge.yaml").exists()
    assert not (ROOT / "deploy-config/profiles/infer_deploy.public.yaml").exists()


def test_prod_image_uses_runtime_stage_without_build_toolchain() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    prod = dockerfile.split("FROM python:3.12-slim AS prod", 1)[1]

    assert "COPY --from=base /app/.venv /app/.venv" in prod
    assert "--no-install-recommends ffmpeg libgl1 libglib2.0-0" in prod
    assert "build-essential" not in prod
    assert "python3-dev" not in prod


def test_prod_image_copies_all_deployment_profiles() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY deploy-config/ ./deploy-config/" in dockerfile


def test_prod_image_ships_the_account_operations_cli() -> None:
    """Account operations run against the live database inside the container.

    Shipping the CLI in the image is what makes `make account-reset-password`
    usable on the deployment host, where there is no repository virtualenv.
    """
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    prod = dockerfile.split("FROM python:3.12-slim AS prod", 1)[1]

    assert "COPY deploy/web/ ./deploy/web/" in prod
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "exec -it vqa-infer .venv/bin/python -m deploy.web.manage_account" in makefile


def test_inference_container_mounts_the_deployment_policy() -> None:
    """Profile changes must take effect without rebuilding the image.

    Without this mount the container reads the copy baked into the image, so
    switching `auth.mode` silently keeps the old policy.
    """
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")
    infer = compose.split("  vqa-infer:", 1)[1].split("  vqa-nginx:", 1)[0]
    assert "../../deploy-config:/app/deploy-config:ro" in infer

    # SELinux-enforcing hosts need the label on this bind mount too. It belongs
    # in the single complete topology instead of a copied Podman volume list.
    assert "../../deploy-config:/app/deploy-config:ro,z" in infer


def test_memory_measurement_script_does_not_invalidate_container_dependency_layers() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "scripts/measure_container_memory.py" in dockerignore


def test_dockerignore_is_build_context_policy_not_runtime_mount_policy() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    compose = (ROOT / "deploy-config/compose/docker-compose.yaml").read_text(encoding="utf-8")

    assert "datasets/" in dockerignore
    assert "reports/" in dockerignore
    assert "../../datasets:/app/datasets:z" in compose
    assert "../../reports:/app/reports:z" in compose


def test_makefile_does_not_invalidate_the_shared_dependency_layer() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY pyproject.toml uv.lock requirements.txt ./" in dockerfile
    assert "COPY pyproject.toml uv.lock requirements.txt Makefile ./" not in dockerfile
    assert dockerfile.count("COPY Makefile ./") == 2


def test_default_deployment_profile_was_removed() -> None:
    assert not (ROOT / "deploy-config/profiles/infer_deploy.yaml").exists()
    runtime = (ROOT / "deploy/config.py").read_text(encoding="utf-8")
    assert 'os.getenv("DEEP_VQA_DEPLOYMENT_CONFIG", "").strip()' in runtime


def test_deployment_check_uses_state_aware_service_budgets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert '[ -n "$$health" ] || health=unconfigured' in makefile
    assert 'check_http vqa-infer "$(API_HEALTH_URL)" no 60' in makefile
    assert 'check_http vqa-nginx "$(WEB_HEALTH_URL)" no 30' in makefile


def test_make_pytest_requires_dev_tool_without_resync() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "$(call require_tool,pytest,dev)" in makefile
    assert "uv run --no-sync pytest -q" in makefile


def test_first_account_can_register_from_the_browser_without_prior_login() -> None:
    page = (ROOT / "frontend/login.html").read_text(encoding="utf-8")
    script = (ROOT / "frontend/login.js").read_text(encoding="utf-8")
    api = (ROOT / "deploy/web/api.py").read_text(encoding="utf-8")
    assert 'id="switchModeButton"' in page
    assert 'state.intent === "register"' in script
    assert 'path: state.intent === "register" ? "auth/register" : "auth/login"' in script
    registration = api.split("async def auth_register(", 1)[1].split("async def auth_login(", 1)[0]
    assert "_require_auth(request)" not in registration
    assert "_check_register_rate_limit(request)" in registration


def test_frontend_is_split_and_uses_redirect_login_flow() -> None:
    frontend = ROOT / "frontend"
    for name in ("index.html", "login.html", "styles.css", "app.js", "login.js"):
        assert (frontend / name).is_file()
    index = (frontend / "index.html").read_text(encoding="utf-8")
    app = (frontend / "app.js").read_text(encoding="utf-8")
    login = (frontend / "login.js").read_text(encoding="utf-8")
    assert 'type="module" src="/app.js"' in index
    assert 'id="historySection"' in index
    assert 'id="result" aria-live="polite" hidden' in index
    assert 'window.location.replace("/login.html")' in app
    assert 'window.location.replace("/")' in login
    assert 'credentials: "include"' in app
    assert 'credentials: "include"' in login
    assert "/frontend-settings" in app
    assert "evaluation_history_enabled" in app
    assert 'id="loginForm"' in (frontend / "login.html").read_text(encoding="utf-8")
    assert "dblclick" in app
    assert "reports/iqa-test" in app


def test_frontend_sources_remain_readable_and_dependency_free() -> None:
    frontend = ROOT / "frontend"
    index = (frontend / "index.html").read_text(encoding="utf-8")
    styles = (frontend / "styles.css").read_text(encoding="utf-8")
    app = (frontend / "app.js").read_text(encoding="utf-8")

    assert max(map(len, index.splitlines())) < 240
    assert max(map(len, styles.splitlines())) < 240
    assert "Inter," not in styles
    assert "⬆" not in index
    assert "◎" not in index
    assert 'id="evaluationLoading"' in index
    assert "setEvaluationLoading(true)" in app
    assert 'historyEmpty: "还没有评估记录。"' in app
    assert 'class="auth-language"' in (frontend / "login.html").read_text(encoding="utf-8")
    assert "text-wrap: balance" in styles
    assert _repository_files("package.json") == []


def test_quality_targets_do_not_resolve_or_download_dependencies() -> None:
    """Once the venv exists, repository gates must be deterministic offline."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "uv run --no-sync ruff format --check ." in makefile
    assert "uv run --no-sync ruff check ." in makefile
    assert "uv run --no-sync mypy --explicit-package-bases" in makefile


def test_gpu_setup_resolves_across_the_trusted_primary_and_torch_indexes() -> None:
    """The partial PyTorch index must not shadow newer primary-index packages."""
    setup = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")

    assert "--extra-index-url $TORCH_INDEX" in setup
    assert "--index-strategy unsafe-best-match" in setup
