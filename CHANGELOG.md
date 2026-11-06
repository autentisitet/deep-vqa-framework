# Changelog

All notable changes to Deep-VQA-Framework are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/).

---

## [0.9.6] - 2026-11-06

### Added

- Added `auth.mode: accounts`: email + password registration, scrypt-hashed
  passwords, hashed session tokens, and account-scoped evaluation history and
  artifacts. The login page adapts to the selected mode, and registration failure
  messages are identical for unknown accounts and wrong passwords.
- Added `make benchmark` (`scripts/benchmark_models.py`): re-runs the published
  serving checkpoints over each dataset's held-out test split using the
  repository metric code, reports raw and five-parameter-logistic-mapped
  accuracy, scores the deterministic OpenCV heuristic on the identical samples,
  and records per-item latency, parameter counts, checkpoint sizes, and process memory into
  `reports/benchmark/<utc>.{json,md}`.
- Added a split bilingual browser frontend with login/workspace pages,
  same-origin browser routing, forwarded-HTTPS cookie handling, upload previews,
  feature-map/Grad-CAM comparisons, and frontend feature settings.
- Added bounded, magic-validated upload handling with streaming byte limits,
  read timeouts, executable-header rejection, random 256-bit filenames,
  restrictive permissions, and post-processing cleanup.
- Added startup model loading: IQA and VQA checkpoints load once during
  worker startup and are reused by request handlers instead of being
  deserialized per request.
- Added container memory measurement with
  `scripts/measure_container_memory.py` and `make docker-memory-measure`.
  The report labels peaks as sampled lower bounds and includes host available
  memory and swap usage.
- Added model, upload, logs, and container health deployment contracts,
  upload-security tests, model-loading tests, and repeatable memory-measurement
  reports.
- Added GitHub Dependabot configuration and a CodeQL workflow for repository
  security automation.

### Changed

- Made `make docker-manage` read-only instead of deleting the runtime's
  dangling images during status inspection.
- Added scoped `make docker-purge-train` and `make docker-purge-deploy`
  cleanup targets; both preserve bind-mounted data and the upload volume.
  Kept `docker-purge-all` project-scoped, removing global runtime pruning.
- Added a session authentication challenge to HTTP 401 responses and mapped
  malformed image/video decoder input to HTTP 422 instead of a server failure.
- Defaulted Docker/Podman Compose to CPU, requiring explicit `GPU=1` on hosts
  with a configured NVIDIA runtime; corrected Make help on account secrets.
- Reduced unnecessary training/inference bind mounts and removed host-gateway
  aliases from the inference container while retaining runtime state paths.
- Kept API documentation on the loopback diagnostic port and removed the
  broken public Swagger link; public model summaries expose only checkpoint
  filenames rather than server directory paths.
- Bounded Nginx request lines and headers, and report request-rate and
  connection-limit rejections as HTTP 429 rather than the default 503.
- Replaced the Nginx missing-path homepage fallback with HTTP 404 and
  prevented an implicit redirect from the bare `/api` prefix.
- Preserved the trusted front proxy HTTPS scheme through bundled Nginx so
  secure cookies and Origin validation use the browser-facing protocol.
- Removed the unused cross-origin browser override and CORS support; the
  bundled frontend now always reaches the API on the same origin.
- Disabled implicit trailing-slash HTTP 307 redirects on API routes so an
  unexpected path returns a normal error response.
- Moved public business error codes to a dedicated numeric range starting at `1000`, separate from HTTP response statuses `100-599`, and stopped the frontend from synthesizing business codes from HTTP statuses.
- Consolidated shared deployment defaults into `deploy-config/basic.yaml`;
  mode profiles now contain only authentication and history overrides. Training
  and deployment share YAML loading and recursive merging, with separate schemas.
- Reused media preprocessing and safe checkpoint loading from `src/` across
  training, deployment and offline EDA, removing the EDA dependency on `deploy/`.

- Simplified media decoding to one backend per role: Pillow for image
  read/decode, Decord for video frame sampling, and OpenCV only for CV
  algorithms (sharpness/noise heuristics). The OpenCV video/image fallback
  paths were removed.
- Reorganized the `deploy` package by concern: `deploy/web` (HTTP routes, wire
  schemas, accounts, evaluation store, uploads, domain exceptions) and
  `deploy/inference` (model loading, preprocessing, prediction, technical
  quality, batch CLI), with the shared instance configuration moved to
  `deploy/config.py`. `deploy/core` was removed.
- Consolidated developer documentation under `docs/` instead of splitting it
  between `docs/` and `knowledge/`, and pointed the frontend footer at the
  repository documentation. The previous footer link resolved to
  `/knowledge/deployment.md` under the nginx web root, which was never readable
  on SELinux-enforcing hosts, so it silently served the application shell.
- Mounted `deploy-config/` into the inference container: profile and Compose
  policy changes now take effect after recreating the container instead of
  requiring an image rebuild. Runtime bind mounts carry their SELinux labels
  in the shared base topology, so CPU overlays do not duplicate volume lists.
- Grouped the test suite by concern (`tests/api`, `tests/deployment`, `tests/ml`,
  `tests/data`) and moved the account operations CLI into the `deploy` package so
  it ships inside the inference image instead of requiring a repository `.venv`.
- Renamed the `cache_clean` Make target to `cache-clean` so every user-facing
  target uses the same hyphenated convention, and documented that GNU Make
  options remain under `make --help`.
- Reworked browser access so network clients connect to Nginx. FastAPI remains
  on the Compose bridge network and its host port binds to `127.0.0.1` by default.
- Standardized container-only browser access, renamed the no-storage
  reverse-proxy mode from `public` to `proxy`, and replaced host `batch-test`
  with `docker-batch-test`.
- Made Compose manage persistent model bind mounts and the upload volume, bounded JSON-file
  container logs, `restart: unless-stopped`, startup health checks, Docker and
  Podman differences, GPU/CPU configuration files, and SELinux-compatible mounts.
- Kept internal mode behind per-person accounts with SQLite evaluation history,
  while proxy mode does not save history or visualization files.
- Hardened generated-file downloads with authentication, path checks, and
  symlink checks, and synchronized generated OpenAPI, mode configuration, Nginx,
  frontend, README, knowledge-base, and contributor documentation.
- Removed local `vuln-audit`, `sbom`, `safety`, and `security-all` Make targets
  and their optional scanner dependencies. Repository security is documented
  around GitHub CodeQL, Dependabot, dependency review, secret scanning, CI,
  and `SECURITY.md` instead of presenting local scanners as a project feature.

### Security

- Scoped generated artifacts and evaluation history to their owner: the account
  in accounts mode, and an anonymous server-issued session in the legacy
  `auth.mode: api_key` deployment where every visitor shares one key. Artifact
  identifiers are random tokens instead of timestamps, and the ownership record
  stores only a digest of the session token.
- Added a bounded artifact retention sweep: `viz-*` directories older than
  `artifacts.retention_seconds` are removed when the next visualization is
  created. Cleanup never runs during startup, and unrelated files under the
  reports root are ignored.
- Exposed only Nginx to network clients and kept inference/model services
  private to the Compose network.
- Added canonical file-type detection from magic bytes instead of trusting
  multipart `Content-Type` or filename extensions.
- Added random upload paths, `0600` file permissions, executable-header
  rejection, bounded request bodies, read timeouts, and cleanup after use.
- Added safe checkpoint loading with `weights_only=True` and startup model
  validation against typed YAML configuration.
- Added bounded container log rotation and GitHub security automation.

### Removed

- Removed the optional Ollama/llama.cpp visual-explanation service, its prompt,
  Compose service, mode configuration, environment variables, API endpoint, frontend
  panel, tests, and documentation. The explanation endpoint reported a
  deterministic OpenCV technical score as an `llm_score` alias and applied a
  Beta-prior "credible interval" to a deterministic value; the service is out of
  scope and did not fit the supported deployment targets.
- Removed the duplicate Chinese disclaimer; `DISCLAIMER.md` remains the
  canonical legal and usage-boundary document.
- Removed the old public-mode YAML file and direct host inference path.
- Removed local SBOM/vulnerability-scanner command interfaces and dependency
  extras that were not part of the maintained runtime contract.

### Not included in v0.9.6

- Redis, Celery, RabbitMQ, Umami, Plausible, and Uptime Kuma are not runtime
  dependencies of this tree. They require separate asynchronous-task,
  monitoring, or analytics designs and are intentionally not represented as
  unused placeholder services.

---

## [0.9.0] - 2026-08-23

### Added

- Added optional containerized Ollama subjective-quality assessment with
  `qwen2.5vl:3b`, a project `Modelfile`, and Bayesian score reporting.
- Added SQLite-backed frontend evaluation history with preview and JSONL export.
- Added Pydantic runtime endpoint configuration shared with `.env` and Compose.
- Added image and video enlarged preview support in the browser frontend.
- Added stable and experimental media-extension classifications, with actual
  OpenCV/Decord decoding required instead of trusting filename suffixes.
- Added pytest coverage for runtime configuration, deployment contracts, media
  extension contracts, SQLite storage, Ollama responses, and visualization.
- Added Compose/Nginx deployment validation and version-consistency checks to CI.
- Added bilingual Ollama guides and expanded command-specific `make help-*`
  documentation.
- Added explicit `train-config/` and `deploy-config/` ownership boundaries for
  training YAML, deployment YAML files, Compose, Nginx, and Ollama files.
- Added a unified `make docker-infer` entry point with `DEPLOYMENT_MODE`.
- Added an explicit backend bridge network for inference and optional Ollama
  services.

### Changed

- Promoted the project version to 0.9.0 and made `pyproject.toml` the source used
  to synchronize runtime and README version markers.
- Centralized API, web, Ollama, and SQLite settings while keeping host and
  container ports distinct.
- Updated Docker/Podman lifecycle commands so `docker-stop` includes Ollama and
  `docker-purge-all` removes the optional Ollama model volume.
- Hardened Docker/Podman networking, proxy handling, localhost-only default port
  bindings, Nginx proxy health checks, and Podman SELinux mounts.
- Renamed deployment services consistently to `vqa-nginx` and `vqa-ollama`.
- Removed the separate Ollama initializer service; model setup now runs as a
  temporary task from the `vqa-ollama` image.
- Changed Ollama model initialization to an explicit non-interactive command for
  Docker/Podman compatibility.
- Updated the frontend, deployment, contributor, and repository-agent documentation
  for the current 0.9.0 workflow.
- Removed obsolete accessibility, cloud-rental, and legacy deployment guide
  documents; the `docs/` documents are now the deployment documentation
  entry point.

### Security

- Kept Ollama model-management endpoints unavailable through Nginx;
  browsers access subjective assessment only through FastAPI.
- Added explicit container-reachable Ollama proxy variables instead of passing
  host loopback proxy addresses into containers.
- Added `.env` to Git and Docker ignore rules and documented safe local binding
  defaults.
- Rejected forged or misleading media suffixes when file content cannot be
  decoded by the configured media decoder.

---

## [0.7.5] - 2026-08-21

### Added

- Added a versioned FastAPI inference API under `/v1`, including health,
  model metadata, evaluation, and visualization resources.
- Added generated OpenAPI, Swagger UI, and ReDoc documentation, with the
  schema persisted to `docs/openapi.json`.
- Added offline training feature-distribution analysis with standardized
  embeddings, PCA files, nearest-sample context, and projection reports.
- Added feature-map and Grad-CAM visualization support for image inference.
- Added registered preprocessing, EDA, and visualization components to make
  extension points explicit and auditable.
- Added configurable per-sample error manifests with sample IDs, MOS values,
  signed/absolute/relative errors, and threshold-based error levels.
- Added bilingual deployment guides, a security policy, an accessibility
  checklist, and updated project disclaimers.

### Changed

- Reworked the browser frontend for image/video upload, automatic IQA/VQA
  routing, model results, JSONL history, and interpretability outputs.
- Stored frontend evaluation history as append-only JSONL records containing
  timestamp, filename, SHA-256 file hash, task type, model, MOS score, MOS
  interval, and inference latency.
- Improved dataset integrity checks for missing, corrupted, duplicate, and
  filename/label-mismatched samples.
- Improved the Docker/Podman workflow with separate API-only and full-stack
  modes, health-gated startup, proxy propagation, persistent OpenAPI output,
  and SELinux-compatible Podman mounts.
- Updated the deployment image to start Uvicorn directly and include runtime
  utilities, visualization modules, and dataset configuration.
- Updated the README documentation and references for the current IQA/VQA
  pipeline, deployment modes, dataset roles, and cited research.

### Security

- Upgraded `cryptography` from 49.0.0 to 50.0.0 to address the vulnerable
  PKCS#7/RSA decryption behavior reported by Dependabot.
- Added `cryptography>=50.0.0` as an explicit project dependency and refreshed
  `uv.lock` and `requirements.txt`.
- Restricted the GitHub Actions workflow token to `contents: read`.
- Documented the API's default deployment security boundaries, including the
  need for authentication, authorization, rate limiting, malware scanning,
  and TLS before exposure to untrusted networks.

---

## [0.7.0] - 2026-08-16

### Changed

- Refined the Pydantic schema and YAML model configuration to match fields consumed
  by the runtime, including early stopping, checkpoint selection, and AMP.
- Reduced checkpoint retention to the two selected best checkpoints per fold.
- Throttled progress updates to avoid recording every one-percent change.
- Improved deployment task selection and batch inference integration.

### Added

- Added Nginx-backed container inference routing.
- Added dated cleanup for old `.pt`, `.csv`, and `.log` files under `results/`.
- Expanded CI with a dedicated Ruff and Mypy job using the locked development
  environment.

### Fixed

- Deployment now keeps task boundaries explicit: IQA accepts images and VQA accepts
  videos; cross-modal conversion is no longer performed.
- Checkpoint loading now restores model configuration and MOS range metadata.
- Unknown configuration keys are rejected instead of being silently accepted.

### Removed

- Removed obsolete dataset schema fields and legacy cross-modal inference helpers.

---

## [0.6.4] - 2026-08-13

### Added

- Added normalized dataset registry handling.
- Added integrity auditing with rejected-label backups, MOS normalization, EDA plots,
  fold metrics, residual plots, and comparison summaries.
- Added modular IQAVQANet components, configurable backbones, losses, metrics, and
  torchvision weight-cache management.
- Added unified inference runtime behavior for API and CLI checkpoint loading.

### Changed

- Improved the deployment container configuration and checkpoint loading path.
- Refreshed dependency lockfiles for the 0.6.4 release.

### Fixed

- Fixed pretrained-backbone input preprocessing so model inputs match the expected
  ImageNet resize and normalization pipeline.
- Fixed and completed leakage-safe group-based train/validation/test splitting and
  group-aware K-fold cross-validation.
- Fixed the training, evaluation, checkpoint selection, and checkpoint deployment
  flow around cross-validation runs.

---

## [0.6.2] - 2026-08-07

### Added

- Added Hatchling packaging support for building the deployment package.
- Added split inference core helpers and a batch inference CLI.
- Added batch test targets and container purge commands.

### Changed

- Hardened inference startup and prevented checkpoint loading from unexpectedly
  downloading pretrained backbones.
- Split environment bootstrap from Python dependency setup.

---

## [0.6.0] - 2026-08-03

### Added

- Added Docker/Podman containerized development, training, and inference deployment
  workflows.
- Added the initial Makefile structure for setup, data preparation, quality checks,
  security checks, and container operations.

### Changed

- Migrated configuration from the previous YAML/DSL approach to Pydantic-backed
  configuration.
- Split system bootstrap from project environment setup.

---

## [0.5.1] - 2026-07-30

### Changed

- Replaced the previous YAML plus DSL configuration flow with typed Pydantic
  configuration and centralized loading.
- Updated training, data, evaluation, and path handling to use the new configuration
  system.

---

## Earlier History

Earlier releases introduced the FastAPI multi-model inference service, frontend
quality-assessment views, MOS denormalization, dataset EDA and integrity checks,
shell-script automation, CI validation, and the original YAML-based training flow.
The complete implementation history is available in `git log`.

### Historical version-label note

Commit `1923cd4` from 2026-05-24 used the message `0.9.2-beta`. That label was an
early, overly optimistic estimate of project maturity made before the current
release process was established. There is no corresponding `v0.9.2-beta` tag,
and it is not treated as a published release or as evidence that later 0.x
versions are regressions. Tagged releases and this changelog are the authoritative
version history.

[0.9.6]: https://github.com/autentisitet/deep-vqa-framework/compare/v0.9.0...59f1112
[0.9.0]: https://github.com/autentisitet/deep-vqa-framework/compare/v0.7.5...v0.9.0
[0.7.5]: https://github.com/autentisitet/deep-vqa-framework/compare/v0.7.0...v0.7.5
[0.7.0]: https://github.com/autentisitet/deep-vqa-framework/releases/tag/v0.7.0
[0.6.4]: https://github.com/autentisitet/deep-vqa-framework/commit/a43df487bee241484d50f7ea3e526f662ba4ba72
[0.6.2]: https://github.com/autentisitet/deep-vqa-framework/commit/dd1a191
[0.6.0]: https://github.com/autentisitet/deep-vqa-framework/commit/77fd572
[0.5.1]: https://github.com/autentisitet/deep-vqa-framework/commit/3737aec
