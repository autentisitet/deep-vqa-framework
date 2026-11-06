# Cross-file contracts

Each row is a surface that changes together. A change that touches one column
and not the rest is an incomplete change.

## Behaviour -> files

| Behaviour | Files that must stay in sync |
| --- | --- |
| Auth mode, accounts, sessions | `deploy-config/basic.yaml` + `deploy-config/profiles/*.yaml`, `deploy/config.py`, `deploy/web/accounts.py`, `deploy/web/api.py`, `frontend/login.js`, `Makefile` (`run_infer_mode`), `tests/api/test_accounts.py`, `docs/frontend.md` |
| Evaluation store | `deploy-config/basic.yaml` + `deploy-config/profiles/*.yaml`, `deploy/config.py`, `deploy/web/evaluation_store.py`, `deploy/web/api.py`, `tests/api/test_frontend_evaluation_store.py` |
| Upload limits and media types | `deploy-config/basic.yaml` + `deploy-config/profiles/*.yaml` (`uploads`), `deploy/web/media_upload.py`, `deploy-config/nginx/default.conf` (`client_max_body_size`), `tests/api/test_media_upload.py` |
| Artifact ownership and retention | `deploy/web/api.py`, `deploy-config/basic.yaml` + `deploy-config/profiles/*.yaml` (`artifacts`), `docs/frontend.md`, `tests/api/test_session_isolation.py` |
| Model checkpoints and MOS range | `deploy-config/basic.yaml` + `deploy-config/profiles/*.yaml` (`models`), `deploy/inference/model_loader.py`, `docs/deployment.md`, `tests/deployment/test_model_loading.py` |
| Business errors and public codes (>=1000, distinct from HTTP status) | `deploy/web/exceptions.py`, `deploy/web/accounts.py`, `deploy/web/api.py`, `deploy/web/schemas.py`, `docs/openapi.json`, `frontend/app.js`, `frontend/login.js`, `docs/frontend.md`, `tests/api/test_errors.py` |
| HTTP routes and schemas (same-origin; no CORS) | `deploy/web/schemas.py`, `deploy/types.py`, `src/data/dataset_types.py`, `deploy/web/api.py`, `docs/openapi.json`, `frontend/app.js`, `frontend/login.js`, `deploy-config/compose/docker-compose.yaml`, `tests/api/` |
| Compose services, mounts, ports | `deploy-config/compose/docker-compose.yaml` plus each overlay, `.dockerignore`, `Dockerfile`, `Makefile`, `tests/deployment/test_deployment_contract.py`, `docs/deployment.md` |
| Nginx routing, headers, and static-file 404 fallback | `deploy-config/nginx/*.conf`, `frontend/index.html`, `tests/deployment/test_documentation_links.py` |
| Public Make targets | `Makefile`, `docs/README.md`, `README.md`, `README_zh.md` |
| Release version | `pyproject.toml`, `src/__init__.py`, `uv.lock`, `requirements.txt`, `README.md`, `README_zh.md`, `DISCLAIMER.md`, `CHANGELOG.md`, `docs/`, then `make version-sync` and `make version-check` |
| Internal type and callable contracts | `pyproject.toml` (Ruff and mypy), `src/core/`, `src/utils/logging_utils.py`, `deploy/inference/predictor.py`, `src/visualization/feature_visualizer.py`, `scripts/`, `tests/ml/`, `tests/api/test_prediction_types.py`, `docs/README.md` |
| Test placement | `tests/<area>/`, `tests/conftest.py` at the root, repository root resolved as `Path(__file__).resolve().parents[2]` |

---

## Configuration precedence

- `.env` at the repository root is loaded with `setdefault`: an exported
  variable always wins over the file.
- Deployment policy (auth mode, store backend, retention, limits) lives in the
  merged `deploy-config/basic.yaml` and profile selected by
  `DEEP_VQA_DEPLOYMENT_CONFIG` (`internal`, `proxy`, or `local`). Missing or
  invalid YAML is a startup error, not a fallback. Nested mappings merge; lists,
  scalars and nulls replace values. Both loaders use `src/config/yaml.py`, then
  validate against their own schemas. Each deployment profile explicitly
  declares `auth.mode`; an absent mode is a startup error.
- Machine-specific values (paths, ports, secrets) come from the environment.
- `deploy/config.py` models use `ConfigDict(extra="forbid")`; an unknown key is
  an error, not a warning.

---

## Security invariants

- Passwords: `hashlib.scrypt` with a per-record salt, stored as
  `scrypt$n$r$p$salt$hash`.
- Session tokens: 256-bit random, stored only as SHA-256 digests.
- Unknown account and wrong password return the same response after the same
  password work, so timing cannot enumerate accounts.
- Artifacts are scoped to `user_id` in accounts mode and to the anonymous
  session digest in legacy mode; a cross-owner read returns `404`, not `403`.
- Generated artifacts expire only when a new one is created, never at startup.

---

## Boundaries

- `deploy/web/api.py` holds HTTP concerns; persistence lives in the store
  classes; model work lives in `deploy/inference/`. `deploy/config.py` is the
  shared instance configuration both sides read.
- `deploy/` may import `src/`; `src/` must not import `deploy/`. Shared safe
  checkpoint loading lives in `src/utils/checkpointing.py`; shared media
  decoding, frame sampling, padding and tensor transforms live in
  `src/data/preprocessing.py`. Deployment wrappers only supply serving defaults
  and checkpoint selection. Training and offline EDA use the shared code directly.
- CLI parsing converts Namespace objects to project-owned options before calling
  execution code; simple commands use explicit standard-library values.
  Numerical interfaces retain Tensor/ndarray semantics, and database connections
  remain an implementation detail of storage rather than an API-layer helper.
- `scripts/` is host-side and needs the repository `.venv`; `deploy/` is
  image-side and runs inside `vqa-infer`.
- Training YAML belongs in `train-config/`; deployment policy belongs in
  `deploy-config/`.

Shared closed choices use `AuthMode` in `deploy/types.py` and `TaskType` /
`DatasetType` in `src/data/dataset_types.py`. These string enums retain their
JSON string values across training configuration, deployment configuration, and
HTTP schemas. The configuration-only evaluation-store backend uses `Literal`.
History responses use `FrontendEvaluationRecord`, excluding stored session IDs;
`mos_interval` remains a JSON array containing exactly two numbers.
