# Deployment Guide / 部署指南

中文版本：[deploy/GUIDE_zh.md](GUIDE_zh.md)

This guide is the single reference for running the Deep-VQA deployment stack.
The root README keeps only a short overview and links here.

## 1. Prerequisites / 前置条件

From the repository root, verify the model checkpoints:

```text
deploy/iqa-models/tid2013_best.pt
deploy/vqa-models/konvid-1k_best.pt
```

The service reads the model configuration and MOS range from each checkpoint.
If neither checkpoint is available, startup fails intentionally.
Review [DISCLAIMER.md](../DISCLAIMER.md) before redistributing or
commercially deploying any checkpoint.

## 2. Choose a mode / 选择运行模式

| Mode | Command | Use case | Entry point |
| --- | --- | --- | --- |
| Host API | `uv run python -m uvicorn deploy.api:app --host 127.0.0.1 --port 8000` | Local API/model debugging | `/v1/*` |
| Container API | `make docker-api` | Containerized API debugging, no frontend | `http://127.0.0.1:8001/v1/*` |
| Full stack | `make docker-infer` | Demo and delivery with frontend + Nginx | `http://127.0.0.1:8000/` and `/api/v1/*` |

`docker-api` starts only FastAPI. `docker-infer` builds the image, waits for
FastAPI health, starts Nginx, and verifies the proxied health endpoint.

## 3. Host API mode / 主机直运行

```bash
uv run python -m uvicorn deploy.api:app --host 127.0.0.1 --port 8000
```

Use these direct paths:

```text
Health:  http://127.0.0.1:8000/v1/health
Models:  http://127.0.0.1:8000/v1/models
Swagger: http://127.0.0.1:8000/docs
ReDoc:   http://127.0.0.1:8000/redoc
```

This mode does not serve the static frontend. For a local frontend preview,
run `python -m http.server 5500 -d frontend` in another terminal and configure
the frontend API base as `http://127.0.0.1:8000/v1`.

## 4. Container modes / 容器运行

### API only

```bash
make docker-api
curl --noproxy '*' http://127.0.0.1:8001/v1/health
```

### Full frontend and API

```bash
make docker-infer
curl --noproxy '*' http://127.0.0.1:8000/api/v1/health
```

The full stack is:

```text
Browser :8000 → Nginx → vqa-infer:8000 → FastAPI → IQA/VQA checkpoint
```

Use `make help-docker-api` and `make help-docker-infer` for command-specific
help. Use `make docker-stop` to stop services and `make docker-purge-all` to
remove project containers and temporary container resources.

## 5. Docker and Podman / Docker 与 Podman

The Makefile detects the available runtime. Do not create a `docker=podman`
alias when using `make docker-*`.

```bash
# Docker
docker compose -f docker/docker-compose.yaml -f docker/docker-compose.docker.yaml up -d vqa-infer nginx

# Podman
podman-compose -f docker/docker-compose.yaml -f docker/docker-compose.podman.yaml up -d vqa-infer nginx
```

The Podman overlay adds `:Z` to Nginx bind mounts for SELinux hosts. This is
also usable on Ubuntu and other non-SELinux Podman installations.

## 6. API resources / API 接口

Through Nginx, use the `/api` prefix:

```text
GET  /api/v1/health
GET  /api/v1/models
GET  /api/v1/models/{model_id}
POST /api/v1/evaluations?model_id=iqa|vqa
POST /api/v1/visualizations?model_id=iqa
```

The evaluation endpoint accepts one multipart field named `file`. The
visualization endpoint accepts images only and returns feature-map/Grad-CAM
paths under `results/diagnostics/`.

For direct host/API-only mode, remove `/api` from each path.

## 7. Batch CLI / 批量推理

```bash
uv run python -m deploy.cli -i examples/images/
uv run python -m deploy.cli -i examples/videos/
uv run python -m deploy.cli -i examples/images/sample.jpg --visualize
```

## 8. Proxy and SELinux / 代理与 SELinux

Compose forwards upper- and lowercase proxy variables and extends
`NO_PROXY`/`no_proxy` with localhost and service names. This prevents health
checks and container-to-container requests from going through a host proxy.

If a host proxy is required during image build:

```bash
make docker-train BUILD_ARGS='--build-arg USE_BUILD_PROXY=true'
```

If Nginx cannot read `default.conf` on Fedora/RHEL, recreate the stack so the
Podman `:Z` mount is applied:

```bash
podman-compose -f docker/docker-compose.yaml -f docker/docker-compose.podman.yaml down
make docker-infer
```

## 9. Troubleshooting / 排障

```bash
podman ps -a
podman logs vqa-infer
podman logs vqa-nginx
podman inspect vqa-infer --format '{{.State.Status}} {{.State.Health.Status}}'
curl --noproxy '*' -v http://127.0.0.1:8000/api/v1/health
```

Interpretation:

- `vqa-infer healthy`, `vqa-nginx healthy`: full stack is ready.
- `vqa-infer healthy`, Nginx `Created`: start/dependency or SELinux mount issue.
- API works on port `8001` but Nginx returns `502`: inspect Nginx logs and service name/network.
- `loaded_models` is empty: checkpoint paths or mounts are incorrect.
