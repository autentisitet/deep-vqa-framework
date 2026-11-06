# Deployment Guide

Deep-VQA has one supported way to start the web service: Docker or Podman.
Host-side Python remains useful for linting, tests, training utilities, and
repository maintenance, but direct host FastAPI deployment is not a supported
deployment method.

## Configuration files

- `deploy-config/basic.yaml` defines shared models, upload limits, SQLite
  limits, retention, and inference defaults.
- `deploy-config/profiles/` overrides authentication and history storage for
  each mode.
- `.env` owns machine-specific ports, bind addresses, secrets, and proxies.
- `deploy-config/compose/` defines the containers and their network connections.

Configuration is loaded through Pydantic with `extra="forbid"`; unknown YAML
fields fail startup.

The loader first reads [basic.yaml](../deploy-config/basic.yaml), then recursively
merges the profile selected by `DEEP_VQA_DEPLOYMENT_CONFIG`. Nested mappings
merge; scalar values, lists, and explicit nulls replace the base value. The
merged result is validated before use. Each profile must explicitly declare
`auth.mode`; an empty override cannot silently disable authentication. Both
files must exist and contain YAML
mappings; a missing or invalid base/profile is a startup error. Custom profiles
use the same layout: `basic.yaml` next to the `profiles/` directory. This uses
the same merge utility as training's `train-config/basic.yaml` and model YAML,
while training and deployment keep separate schemas and policy files.

The selected configuration tree is mounted into `vqa-infer` read-only, so
editing a profile takes effect after the container is recreated; the image does
not have to be rebuilt. The image also contains a copy, which applies only to
deployments that do not mount that directory.

---

## Mounts and runtime isolation

Compose shares one base topology across Docker and Podman; the CPU/GPU overlays
only express accelerator differences. The long mount list includes **optional
training and development containers**, not only the web service:

| Service | Host/volume inputs | Why they stay |
| --- | --- | --- |
| `vqa-nginx` | `frontend/` and two Nginx configuration files (read-only) | Serve static assets and forward `/api/` inside the Compose `backend` bridge network. |
| `vqa-infer` | Two model directories (read-only), `deploy-config/` (read-only), `examples/` (read-only), `reports/`, `docs/`, `.cache/`, and the named `vqa-uploads` volume | Preloaded model weights, selected policy, optional container batch tests, accounts/history/artifacts, generated OpenAPI at startup, model cache, and temporary uploads respectively. It does not need the training `results/` directory. |
| `vqa-train` | Datasets, results, reports, model output directories, and `.cache/` | Train and export selected checkpoints; it does not need `docs/`. |
| `vqa-dev` | Source, configuration, scripts, tests, datasets, results, reports, and `.cache/` | Interactive development; these are not web-service permissions. |

Only the Nginx port `127.0.0.1:8000` is a supported browser entry. The
Uvicorn diagnostic port `127.0.0.1:8001` bypasses Nginx limits; both must remain
loopback-bound. Inside the bridge network Nginx reaches `vqa-infer:8000` by its
Compose service name (no host-gateway alias or host-port hairpin). HTTP proxy
environment variables used for outbound downloads do not route this internal
hop; if custom proxy settings are introduced, exclude the service name from
proxying. A syntax-valid Compose merge does not prove either runtime's DNS,
SELinux labels, GPU access, mounts, or actual container networking: verify
these on the deployment host before exposing the stack.

---

## Network connections

```text
Browser -> 127.0.0.1:8000 -> vqa-nginx:80
  -> /api/* -> vqa-infer:8000 (Uvicorn ASGI -> FastAPI /v1/*)
       ├─ IQA checkpoint
       └─ VQA checkpoint
```

| Service | Default image | Required | Purpose |
| --- | --- | --- | --- |
| `vqa-infer` | `vqa-infer:latest` | Yes | FastAPI, IQA/VQA inference, generated files |
| `vqa-nginx` | `nginx:alpine` | Yes | Static frontend; forwards `/api/` to FastAPI |
| `vqa-train` | `vqa-train:latest` | No | Training workflow |
| `vqa-dev` | `vqa-dev:latest` | No | Reproducible development shell |

All active services share the private `backend` bridge network. Only localhost
ports are published by default:

```text
8000 -> Nginx/browser entry
8001 -> Uvicorn/FastAPI loopback-only diagnostics (bypasses Nginx)
```

---

## Prerequisites

Serving checkpoints must exist at:

```text
deploy/iqa-models/tid2013_best.pt
deploy/vqa-models/konvid-1k_best.pt
```

Internal mode has no shared secret to create. The first visitor registers an
account in the browser. Account records and evaluation history are stored in
separate SQLite files under `reports/` by default; configure them independently
with `DEEP_VQA_ACCOUNTS_SQLITE_PATH` and `DEEP_VQA_SQLITE_PATH`. The reports
directory must be persistent if accounts or history must survive container
recreation. `make env-secrets` only matters for the legacy `auth.mode: api_key`
deployment.

For local maintenance, account state can be changed without exposing an HTTP
administration endpoint:

```bash
make account-disable EMAIL=user@example.com
make account-enable EMAIL=user@example.com
make account-reset-password EMAIL=user@example.com
```

Disabling an account and resetting its password revoke active sessions. The
password command reads the new password interactively and does not accept it as
a Make variable or command-line argument.

The supported browser/API request path goes through the bundled Nginx. The
loopback-only diagnostic port 8001 reaches Uvicorn directly and bypasses Nginx
request limits; do not publish this port on a private or public IP, even behind
a front proxy. In proxy mode an additional front process handles authentication
and TLS before forwarding to the bundled Nginx port 8000. Serving through
Nginx does not by itself provide TLS or per-user authentication.

---

## Deployment modes

The service is a set of processes on one machine. Who can reach it depends on
the IP address its ports bind to:

- `127.0.0.1` (loopback): only processes on that same machine. Packets never
  leave the machine's network stack, so DNS, NAT, and public IPs are not
  involved.
- A private IP (`192.168.x.x` / `10.x`): reachable from the same LAN.
- A public IP: reachable from the Internet, reached through DNS and NAT.

| Mode | Command | Where the service listens | Login and saved data |
| --- | --- | --- | --- |
| Internal | `make docker-infer` | `127.0.0.1` only | Email + password accounts, SQLite history, per-account files |
| Local | `make local` | `127.0.0.1` only | No login; history scoped to an anonymous browser session |
| Proxy | `make docker-infer DEPLOYMENT_MODE=proxy` | `127.0.0.1` still; a separate process in front owns the reachable address | No app login or saved data; the front process authenticates |
| GPU (opt-in) | `make docker-infer GPU=1` | same as Internal | Requires a working NVIDIA runtime/driver |

`local` is the internal stack with `auth.mode: none`
(`deploy-config/profiles/infer_deploy.local.yaml`). History and artifacts are
scoped to an anonymous browser session cookie, so clearing cookies loses the
history view. Both bind only `127.0.0.1`; never publish them to a network.

### How proxy mode reaches the app

Proxy mode (`auth.mode: none`, `evaluation_store.backend: none`) does not put
the app on a public IP. A separate process you operate owns the reachable
address, terminates TLS, authenticates users, then forwards requests to the
bundled Nginx loopback port:

```text
browser -> example.com -> DNS -> public IP -> NAT maps :443 to a front process
        -> the front process does HTTPS + login
        -> forwards to bundled Nginx on 127.0.0.1:8000
        -> Uvicorn on the private Compose network -> FastAPI /v1/*
```

The browser frontend is served only through bundled Nginx at the same origin
as `/api/v1/*`. FastAPI does not enable CORS, and the browser does not accept a
cross-origin API-base override. If a separate front process handles TLS and
authentication, forward the entire site (HTML and API) to bundled Nginx so the
browser still uses one origin.

The front process must strip client-supplied forwarding headers and set
`X-Forwarded-Proto: https` after terminating TLS; bundled Nginx preserves that
value for the app so secure cookies and Origin validation use the browser's
scheme. Only send this header through the loopback-bound Nginx entry. The front
process can run on the same machine or reach its loopback interface through a
controlled tunnel. The Uvicorn diagnostic port 8001 must stay on loopback so
network clients cannot bypass Nginx and its request limits.

If the machine running the app is behind NAT with no public IP, the front
process can still reach it through a reverse tunnel (ngrok, frp, cloudflared):
the app machine dials out first and incoming traffic returns over that
connection. The tunnel is external to this repository.

The application and bundled Nginx do not provide DDoS protection. Their rate
and connection limits are local resource controls and cannot absorb volumetric
traffic, upstream bandwidth exhaustion, or a distributed layer-7 attack. For
Internet-facing use, place a reverse proxy, load balancer, or DDoS-filtering
service in front of the host, keep the FastAPI port private, and restrict the
origin so clients cannot bypass it.

There is no supported `host-infer` command. Use the documented Make commands to
start, check, and stop the service.

---

## Startup order and health

`make docker-infer` starts services in this order:

1. build and start `vqa-infer`, which loads both configured checkpoints once;
2. check FastAPI `/v1/health` to confirm that both models loaded;
3. start `vqa-nginx` and verify `/api/v1/health` end to end.

`make docker-deploy-check` requires `vqa-infer` and `vqa-nginx`.

```bash
make docker-deploy-check
make docker-manage
make docker-stop
```

---

## Docker and Podman

The Makefile detects Docker or Podman and adds the matching configuration file.
CPU-only inference is the default. On hosts with a verified NVIDIA runtime,
driver, and GPU, opt in explicitly:

```bash
make docker-infer GPU=1
```

`GPU=0` selects the CPU-only overlay. Both CPU overlays
are intentionally empty because CPU execution has no engine-specific runtime
settings. The base topology carries the bind mounts and their SELinux labels:
`:z` for paths shared by services and `:Z` for Nginx-private files. Docker uses
the same topology; on hosts without SELinux the labels have no policy effect.

`.dockerignore` cannot replace these mount options. It controls which files are
sent to the image builder, while Compose bind mounts are created later when a
container starts. Datasets and reports are intentionally excluded from the
build context and still mounted at runtime. Model directories are direct
read-only bind mounts so their SELinux label is explicit instead of being
hidden behind a local-volume driver.

For manual inspection only, the equivalent Compose configuration is based on:

```bash
docker compose \
  -f deploy-config/compose/docker-compose.yaml \
  -f deploy-config/compose/docker-compose.docker-gpu.yaml \
  config
```

Use the Make commands because they select the correct Docker/Podman and mode
configuration files and wait for the services to become ready.

### Rebuilding after CA store updates

The `certifi` package and the `python:3.12-slim` base image's `ca-certificates`
ship the CA trust store used by outbound HTTPS, such as `gdown` downloading
checkpoints. After a root-store update such as the October 2026 eMudhra
intermediate-CA compromise, refresh both:

```bash
make docker-infer BUILD_ARGS="--pull"
```

`certifi` is pinned through `uv.lock` and `requirements.txt`; run `make setup`
to sync it into the environment.

`nginx:alpine` does no TLS in this project: it listens on port 80 and proxies
to `http://vqa-infer:8000`, so its bundled CA store is never consulted. It
only needs refreshing if a future change adds HTTPS termination or an HTTPS
upstream.

This rebuilds only the `vqa-infer` image. The `vqa-train` and `vqa-dev` images
and the trained checkpoints under `deploy/iqa-models/` and
`deploy/vqa-models/` are not touched. Do not run `make docker-purge-all` for
this update.

---

## Request and response transport

The bundled Nginx-to-Uvicorn upstream explicitly uses **HTTP/1.1**
(`proxy_http_version 1.1`). Browser-to-Nginx traffic is plain HTTP with this
configuration; its negotiated version depends on the client and any external
TLS-terminating front proxy. The API uses ordinary request/response, not a
persistent event channel. Browser `fetch()` sends JSON for account
operations and multipart form-data for media uploads. Nginx does not buffer the
entire proxied upload (`proxy_request_buffering off`); the application reads the
uploaded file in bounded chunks. This is **upload streaming**, not a streamed
inference response: evaluation and visualization complete synchronously and
return one JSON result. Generated PNGs are fetched later with ordinary GET;
`FileResponse` may support HTTP range/partial-content downloads through the
framework, but no live media stream is offered.

There are **no** WebSocket, WebRTC, SSE, webhook, short-polling, long-polling,
or asynchronous job-status endpoints. Refreshing history is an explicit user
action. If inference lasts longer than the configured proxy read timeout, the
proxy may return a gateway timeout; adding 202/polling would require a separate
job lifecycle and persistent state, not just a different status code.

---

## HTTP responses and business codes

HTTP status is part of the **response**, not the request. HTTP 1xx responses
(such as `100 Continue`) are handled by the HTTP server/client when applicable;
the project does not use them as business results. The layers can respond
independently:

| Layer | Representative statuses | Meaning in this deployment |
| --- | --- | --- |
| Nginx | 400, 413, 414, 429, 502, 504 | Malformed/oversized headers, body too large, request URI too long, throttling, unreachable or timed-out upstream. Responses are not guaranteed to use FastAPI JSON. |
| Uvicorn | 400, 100 when negotiated | Protocol parsing or `Expect: 100-continue`; no business error code. |
| FastAPI and route handlers | 200, 201, 401, 403, 404, 405, 413, 422, 429, 500 | Reads/login, created resources, authentication/Origin, routing, upload validation, input validation, login throttling, unexpected failures. A request-time configuration failure can return 503. |

The FastAPI error body has a separate numeric business `code` from 1000 to 1601
(see [API errors](frontend.md#api-errors)); HTTP status remains a valid
100–599 response code. A mismatched method returns 405 with `Allow`. A mistyped
trailing slash returns 404 rather than the framework's implicit 307 redirect.
Browser login uses `window.location.replace` to load a new document; it is not
an HTTP 3xx response. The proxy forwards `/api/v1/*` internally to `/v1/*`
without instructing the client to redirect or resend an upload.
An unknown static path or bare `/api` returns 404 rather than the homepage
or an implicit Nginx redirect.

Nginx bounds an individual request line with an 8 KiB buffer
(`large_client_header_buffers 4 8k`): a line that does not fit returns 414
before FastAPI. Oversized individual headers
are rejected separately; the buffer count is not a 32 KiB URI allowance. Nginx
caps request bodies at 101 MiB, applies `api_requests` (30 requests/minute with
burst 15), `auth_login` (5 requests/minute with burst 5), and at most 10
concurrent API requests per client IP. Excess rate/concurrency now returns 429,
not Nginx's default 503; unlike application login throttling, Nginx does not
promise a `Retry-After` header. Application login/registration windows and
streamed upload size checks are a second layer. These local limits are not DDoS
protection. In proxy mode a trusted front proxy must enforce its own limits and
provide the real client IP; otherwise all clients behind it share Nginx's IP
quota. The loopback-only Uvicorn diagnostic port 8001 bypasses Nginx limits.

The existing `deploy.inference.cli` performs **offline** inference and loads
checkpoints directly; it is not an HTTP client. For a deployed service run
`make docker-deploy-check` to check both the loopback API and the Nginx entry,
then use `curl --noproxy '*'` against `http://127.0.0.1:8000/api/v1/health`
and a representative authenticated upload. `httpx` is not required for this:
putting HTTP traffic into the offline CLI would mix two execution contracts.
The health check alone does not prove authentication, upload, or inference works.


---

## API resources

Through Nginx, use:

```text
GET  /api/v1/health
GET  /api/v1/frontend-settings
GET  /api/v1/models
GET  /api/v1/models/{model_id}
GET  /api/v1/frontend-evaluations
POST /api/v1/evaluations?model_id=iqa|vqa
POST /api/v1/visualizations?model_id=iqa
GET  /api/v1/auth/status
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/logout
GET  /api/v1/files/{path}
```

The browser fetches HTML, JavaScript, CSS, and the SVG logo as ordinary GET
responses from the bundled Nginx static file root (`frontend/`). No object
storage or FastAPI route serves the frontend. Nginx sends only `/api/` requests
to Uvicorn/FastAPI; its `/api/v1/` prefix is replaced internally by `/v1/`,
while the remaining path and query string are preserved. The file endpoint
uses FastAPI's `{file_path:path}` to capture the rest of the URL after
`/v1/files/`; it then validates segments, resolves the path beneath the reports
root, and checks artifact ownership. Do not concatenate an unvalidated user
path into a filesystem path.

`/health` reports IQA/VQA readiness for the configured model catalog.
`/models` lists the configured IQA/VQA models and reports whether each one is
loaded in memory. Its checkpoint field contains only the configured filename,
not a host/container filesystem path. The FastAPI process loads both checkpoints once during startup;
request handlers do not load them again. See [model and container memory](performance.md).
`frontend-settings` reports whether login, history, and visualization are
enabled so the browser can show the matching controls.

The `auth` routes exist in every mode and behave according to the profile. With
`auth.mode: accounts` they verify the email and password with scrypt and exchange
them for a random session token stored as a digest. With
`auth.mode: api_key` they perform the shared-key check and the HttpOnly cookie
exchange. With `auth.mode: none` (proxy mode) `/auth/status` reports
`required: false` and the login call returns immediately, because the reverse
proxy is expected to authenticate users instead.

These routes use nouns for resources and HTTP methods for actions:
`GET /models` lists models, `GET /models/{model_id}` selects one, `POST
/evaluations` and `POST /visualizations` create completed results, and `GET
/frontend-evaluations` reads history. The `auth/login`, `auth/logout`,
`auth/register`, `auth/status`, and `frontend-settings` endpoints are pragmatic
authentication/UI operations, not pure resource CRUD. No route accepts or
executes an arbitrary server-side function. Do not change the existing public
paths just to enforce a naming style without a migration plan.

The FastAPI framework registers diagnostic `/docs`, `/redoc`, and
`/openapi.json` automatically; do not duplicate them with route decorators.
The bundled Nginx blocks `/api/docs`, `/api/redoc`, and `/api/openapi.json`,
and there is no browser UI button for these explorers. An operator on the
instance can inspect the built-in documents via the loopback-only Uvicorn
port, for example `http://127.0.0.1:8001/docs` and `/redoc`, or read the
repository-generated `docs/openapi.json`. API docs are a contract, not an
authorization boundary: private credentials, server paths, undocumented
admin functions, and model weights must never appear in public responses.

`/files/` serves the generated PNG artifacts referenced by a visualization
response. It is omitted from the generated OpenAPI schema
(`include_in_schema=False`) but still requires authentication, rejects paths that
escape the reports directory, and requires the caller to own the artifact. In
accounts mode the owner is the account id, so the artifact follows the person
across browsers. In the legacy `api_key` mode it is the anonymous browser
session, because every visitor there shares one key and authentication alone
would let any visitor read any artifact. In proxy mode it returns 404, because
that mode does not save visualization files.

---

## Account operations

Internal mode has no email delivery, so there is no self-service password
recovery. An operator with shell access can disable, re-enable, or reset an
account:

```bash
make account-disable EMAIL=user@example.com
make account-enable EMAIL=user@example.com
make account-reset-password EMAIL=user@example.com
```

These run inside the inference container, because they modify the live accounts
database and need the service's dependencies. The equivalent without the
Makefile is:

```bash
docker exec -it vqa-infer .venv/bin/python -m deploy.web.manage_account reset-password user@example.com
```

Disabling an account and resetting its password both revoke that account's
existing sessions. The accounts database is `reports/accounts.sqlite3` inside
the container, backed by the mounted `reports/` directory.

---

## Containerized batch inference

Batch IQA/VQA uses the production inference image without starting the browser
stack:

```bash
make docker-batch-test
make docker-batch-test IMAGES=1
make docker-batch-test VIDEOS=1
```

This replaces the former host `batch-test` command and uses the same PyTorch,
FFmpeg, and checkpoint code as the HTTP service.

---

## Proxy settings

Package-download proxy settings use `BUILD_ARGS`. Keep `vqa-infer` and localhost in
`NO_PROXY`/`no_proxy`.

---

## Destructive cleanup

```bash
make docker-stop          # stop project containers; preserve all data
make docker-purge-train   # remove only dev/train containers and images
make docker-purge-deploy  # remove only Nginx/inference containers and inference image
make docker-purge-all     # remove all project containers/images/network/upload volume
```

`make docker-manage` only inspects the selected runtime and does not remove
images belonging to this or other projects.

The scoped targets keep bind-mounted datasets, checkpoints, results, SQLite
accounts/history, and the named upload volume. `docker-purge-all` removes the
project's containers and named upload volume (including its contents) plus
project images and the Compose network. It does **not** run global runtime
volume/network/image pruning or delete any host bind-mounted directory.
Review `make -n docker-purge-all` before invoking it.


---

## Upload and container-log safety

All Compose services use the runtime `json-file` log driver with bounded
rotation. Defaults are configurable in `.env`:

```env
CONTAINER_LOG_MAX_SIZE=50m
CONTAINER_LOG_MAX_FILES=3
```

Uploads are streamed with a hard byte limit and per-read timeout. The API does
not trust multipart `Content-Type`: it identifies supported image/video formats
from file-header magic before persistence. ELF, PE, shebang script, WebAssembly,
and Mach-O executable headers are rejected before a file is created. Accepted
media is written with mode `0600` under a random 256-bit token filename plus a
canonical extension derived from magic. Client filenames are display-only and
never participate in filesystem paths or command execution.

---

## Files saved by each mode

Internal mode uses SQLite for evaluation history and saves visualization PNG
files under `reports/iqa-test/`. Proxy mode uses
`evaluation_store.backend: none`; it does not create SQLite or JSONL history
records and does not provide saved visualization files. Uploaded media is
removed after processing in both modes.
