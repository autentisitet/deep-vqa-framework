# Deployment Guide

This guide is the single reference for running the Deep-VQA deployment stack.
The root README keeps only a short overview and links here.
Run every command below from the repository root; paths such as `.env.example`
and `deploy-config/` are relative to that directory.

### Configuration ownership

The selected deployment profile YAML is the repository-managed policy: upload
limits, SQLite limits, storage backend, authentication mode, and Ollama policy
live there. The root `.env` contains host-specific values that must vary by
machine (bind addresses, published ports, Ollama URL, SQLite path, API key,
and cookie-signing secret). `deploy/core/runtime_config.py` loads both sources,
then validates the resulting typed configuration with Pydantic using
`extra="forbid"`; unknown YAML fields fail startup. Do not duplicate policy
values in `.env`.

## How the container pieces fit together

These terms describe different layers and should not be treated as synonyms:

| Layer | In this repository | Meaning |
| --- | --- | --- |
| Image | `vqa-infer`, `vqa-dev`, `vqa-train`, `nginx:alpine`, `ollama/ollama:latest` | A build artifact or downloaded package. The custom images are built from the corresponding Dockerfile stage; Nginx and Ollama use official images. |
| Compose service | `vqa-infer`, `vqa-nginx`, `vqa-ollama`, `vqa-dev`, `vqa-train` | A declarative runtime definition: image/build target, mounts, environment, ports, health checks, and network membership. |
| Container | Usually the same name as the service (`vqa-infer`, `vqa-nginx`, `vqa-ollama`) | A running or stopped instance created from an image and a service definition. `vqa-dev` and `vqa-train` are normally short-lived interactive containers. |
| Network | Compose network `backend` (bridge) | Private container-to-container DNS and connectivity. Services can reach `http://vqa-infer:8000` and `http://vqa-ollama:11434` by service name. |
| Published port | Host `8001`, `8000`, and optional `11434` | A host-side entry point mapped to a container port. Defaults bind to `127.0.0.1`; they are not the private service DNS addresses. |

The normal inference path is:

```text
Browser :8000
  -> vqa-nginx:80 (published host port 8000)
  -> vqa-infer:8000 (private backend network)
  -> IQA/VQA model checkpoint
```

When subjective assessment is enabled, FastAPI additionally calls
`vqa-ollama:11434` over `backend`. Ollama is not called from inside Nginx and
is not exposed through an `/api` proxy route. The public host port `11434` is
only for local diagnostics or host-mode integrations.

Startup ordering is deliberately split into dependency types:

1. Images are built or pulled first. Building `vqa-infer` and `vqa-dev` shares
   the `base` stage; it does not start another service.
2. `vqa-infer` starts independently and becomes healthy after loading its
   checkpoints. It has no `depends_on` requirement.
3. `vqa-nginx` starts only after the `vqa-infer` health check passes, then its
   own GET probe verifies the proxied route.
4. `vqa-ollama` is optional and independent at the Compose level. The
   `*-ollama` Make target starts it, waits for `/api/tags`, and runs model setup
   as a short-lived task from the same Ollama image before starting inference
   with `OLLAMA_BASE_URL=http://vqa-ollama:11434`.

This is a small single-host service composition, not a Kubernetes-style
microservice platform. The bridge network is intentional: it provides service
name resolution and isolation without the operational cost of overlay,
macvlan, or host networking. `docker-compose.docker.yaml` and
`docker-compose.podman.yaml` only add runtime-specific GPU/SELinux settings;
they do not create a second application topology.

## 1. Prerequisites

From the repository root, verify the model checkpoints:

```text
deploy/iqa-models/tid2013_best.pt
deploy/vqa-models/konvid-1k_best.pt
```

The service reads the model configuration and MOS range from each checkpoint.
If neither checkpoint is available, startup fails intentionally.
Review [DISCLAIMER.md](../DISCLAIMER.md) before redistributing or
commercially deploying any checkpoint.

For an internal deployment, set `auth.mode: api_key` in
`deploy-config/profiles/infer_deploy.internal.yaml` and provide `DEEP_VQA_API_KEY` in `.env`. The same
frontend then shows a login panel; evaluation APIs and generated artifacts are
protected by an HttpOnly cookie. Keep `auth.mode: none` only for trusted local
experiments.

Also set an independent `DEEP_VQA_AUTH_SECRET` for signing the login cookie.
`make bootstrap` installs `openssl` on Debian/Ubuntu systems. Generate both
values with `openssl rand -hex 32`; this is a shared deployment
key flow, with no registration or 2FA account system.

To demonstrate the internal mode, create `.env`, generate the two secrets, and
start the stack:

```bash
make env-secrets
make docker-infer
```

Open `http://127.0.0.1:8000/`. The page displays an **API key** field; paste
the exact value from `DEEP_VQA_API_KEY` and press **Unlock**. The backend sets
an HttpOnly session cookie, after which the workspace and history become
available. There is no registration page because this is deployment-level
shared-key authentication, not a multi-user account service.

## 2. Choose a mode

### Required first-time setup

Inference Compose targets use host-specific values from a root `.env` file. If
it is missing, the Makefile creates it from `.env.example` automatically. The
internal profile still requires you to fill in both secret values before
startup; public mode does not require those secrets.

```bash
cp .env.example .env  # optional; inference targets also create this template
```

The repository policy defaults to `auth.mode: api_key`. For that mode, fill in
both values before starting FastAPI:

```bash
sed -i "s|^DEEP_VQA_API_KEY=.*|DEEP_VQA_API_KEY=$(openssl rand -hex 32)|" .env
sed -i "s|^DEEP_VQA_AUTH_SECRET=.*|DEEP_VQA_AUTH_SECRET=$(openssl rand -hex 32)|" .env
```

If this is only a trusted local demo and no login screen is desired, change
use `deploy-config/profiles/infer_deploy.public.yaml`; the key variables are then
not required.

### Host topology

The compose files are a reference deployment, not a requirement that all
components share one production host:

| Topology | Placement | Storage/auth guidance |
| --- | --- | --- |
| Internal minimum (2 hosts) | Host A: Nginx + static frontend; Host B: FastAPI + model runtime + local SQLite | Keep SQLite on Host B's local disk; restrict FastAPI to Host A's private address and use an API key or enterprise reverse-proxy auth. |
| Public/stateless (2–3 hosts) | Host A: TLS/WAF/reverse proxy; Host B: FastAPI + models; optional Host C: Ollama | Set `evaluation_store.backend: none`; use SSO/OIDC or a gateway-issued identity instead of a shared API key. |
| External enterprise platform | Managed by the organization outside this project | SSO, WAF, multi-tenant identity, and server databases are explicitly out of scope for Deep-VQA-Framework. |

The single-host compose stack is intended for development, demos, and a
single-instance internal workbench. It is not the recommended production
topology. Do not put the SQLite file on NFS or a shared volume accessed by
multiple FastAPI hosts.

The `DEEP_VQA_API_KEY` flow documented below is a bootstrap/service-token
mechanism for a trusted single-instance deployment. It is deliberately not
presented as user registration or identity management. For a real multi-user
internal or public service, terminate authentication at the enterprise edge
(OIDC/SAML/SSO, mTLS, or an API gateway) and pass the authenticated identity to
FastAPI.

| Mode | Command | Use case | Entry point |
| --- | --- | --- | --- |
| Host API | `uv run python -m uvicorn deploy.api:app --host 127.0.0.1 --port 8000` | Local API/model debugging | `/v1/*` |
| Full stack | `make docker-infer` or `make docker-infer DEPLOYMENT_MODE=public` | Demo and delivery with frontend + Nginx | `http://127.0.0.1:8000/` and `/api/v1/*` |

The profile targets build the image, wait for FastAPI health, start Nginx, and
verify the proxied health endpoint.

The Compose bind mount `../.cache:/app/.cache` only makes the host directory
available inside the container; it does not select a library cache path. The
production image therefore sets `XDG_CACHE_HOME` and `TORCH_HOME` so PyTorch
downloads use the mounted directory. `UV_CACHE_DIR` is intentionally omitted
from the serving image because uv is used during build/development only.

### Container state checks

Startup and `make docker-deploy-check` use one state model for every service.
The decision is based on `container state / health state`: `running/healthy`
is verified immediately, `created/*` and `running/starting` receive a bounded
wait, `running/unconfigured` receives two HTTP probes, and
`exited/*`, `dead/*`, `removing/*`, or `*/unhealthy` fail immediately.
`vqa-infer` is required; Ollama is optional. Nginx is optional only when using
the direct API, but the full `make docker-infer` target intentionally starts
and verifies it. The check treats an absent Nginx/Ollama container as optional,
but fails when either is present and unhealthy.

The state vocabulary is identical for every service. Timeout budgets differ
only because model loading can take longer than starting Nginx: FastAPI and
Ollama receive up to 120 seconds, while Nginx receives up to 60 seconds after
the FastAPI check. An empty runtime health value is normalized to
`unconfigured`.

### Deployment matrix

Choose the policy in deployment profiles
before starting. `auth.mode: api_key` requires `DEEP_VQA_API_KEY` in `.env`;
`evaluation_store.backend: none` disables history persistence.

| Scenario | YAML settings | Without Ollama | With Ollama |
| --- | --- | --- | --- |
| Internal | `auth.mode: api_key` + `evaluation_store.backend: sqlite` | `make docker-infer` | `make docker-infer-internal-ollama` |
| Public stateless | `auth.mode: none` + `evaluation_store.backend: none` (external gateway owns auth) | `make docker-infer DEPLOYMENT_MODE=public` | `make docker-infer-public-ollama` |
| Local trusted development | `auth.mode: none` + `evaluation_store.backend: none` | `make docker-infer DEPLOYMENT_MODE=public` | `make docker-infer-public-ollama` |

The profile targets validate the YAML policy and then call the common stack
target. They do not edit configuration files. Ollama is independent:
`*-ollama` prepares `deep-vqa-subjective` and then
starts the same inference stack without changing authentication or storage.

See the [frontend guide](frontend.md) for the shared UI and login flow.

## 3. Host API mode

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

## 4. Container modes

### Full frontend and API

```bash
make docker-infer
curl --noproxy '*' http://127.0.0.1:8000/api/v1/health
```

The full stack is:

```text
Browser :8000 → Nginx → vqa-infer:8000 → FastAPI → IQA/VQA checkpoint
```

Use `make help-docker-dev` for command-specific
help. Use `make docker-stop` to stop services and `make docker-purge-all` to
remove project containers and temporary container resources.

`make docker-infer` is the default user mode and does not start Ollama.
Use `make docker-infer-internal-ollama` for the internal API + Nginx + Ollama
stack, or `make docker-infer-public-ollama` for the public profile.

## 5. Docker and Podman

The Makefile detects the available runtime. Do not create a `docker=podman`
alias when using `make docker-*`.

```bash
# Docker
docker compose -f deploy-config/compose/docker-compose.yaml -f deploy-config/compose/docker-compose.docker.yaml up -d vqa-infer vqa-nginx

# Podman
podman-compose -f deploy-config/compose/docker-compose.yaml -f deploy-config/compose/docker-compose.podman.yaml up -d vqa-infer vqa-nginx
```

The Podman overlay adds `:Z` to Nginx bind mounts for SELinux hosts. This is
also usable on Ubuntu and other non-SELinux Podman installations.

## 6. API resources

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
paths under `reports/iqa-test/`.

The optional `POST /api/v1/subjective-assessments` endpoint uses the local
Ollama model configured in `deploy-config/ollama/Modelfile`. It accepts images and videos,
samples up to four video frames by default, and returns a quality description,
the model score, and a Bayesian posterior. It does not write evaluation logs.
See the [Ollama guide](ollama.md) to start Ollama and create the
`deep-vqa-subjective` model.

Uploads are copied with the limits in
deployment profiles: a 100 MiB default limit
and a 30-second read timeout. Frontend evaluation SQLite uses the same file
for its 256 MiB database-size cap and 5-second busy timeout. Edit the YAML to
change this shared policy. Set `evaluation_store.backend: none` for a stateless
public deployment. Host-specific endpoints, ports, proxies, and the optional
SQLite path remain in `.env`.

For direct host/API-only mode, remove `/api` from each path.

## 7. Batch CLI

```bash
uv run python -m deploy.cli -i examples/images/
uv run python -m deploy.cli -i examples/videos/
uv run python -m deploy.cli -i examples/images/sample.jpg --visualize
```

## 8. Proxy and SELinux

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
podman-compose -f deploy-config/compose/docker-compose.yaml -f deploy-config/compose/docker-compose.podman.yaml down
make docker-infer
```

## 9. Troubleshooting

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
