# Changelog

All notable changes to Deep-VQA-Framework are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/).

## [0.7.5] - 2026-08-21

### Added

- Added a versioned FastAPI inference API under `/v1`, including health,
  model metadata, evaluation, and visualization resources.
- Added generated OpenAPI, Swagger UI, and ReDoc documentation, with the
  schema persisted to `docs/openapi.json`.
- Added offline training feature-distribution analysis with standardized
  embeddings, PCA artifacts, nearest-sample context, and projection reports.
- Added feature-map and Grad-CAM visualization support for image inference.
- Added registered preprocessing, EDA, and visualization components to make
  extension points explicit and auditable.
- Added configurable per-sample error manifests with sample IDs, MOS values,
  signed/absolute/relative errors, and threshold-based error levels.
- Added bilingual deployment guides, a security policy, an accessibility
  checklist, and updated project disclaimers.

### Changed

- Reworked the browser frontend for image/video upload, automatic IQA/VQA
  routing, model results, history export, and interpretability outputs.
- Frontend evaluations are appended to
  `reports/frontend-evaluations.jsonl` with timestamp, filename, SHA-256 file
  hash, task type, model, MOS score, MOS interval, and inference latency.
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

## [0.7.3] - 2026-08-20

### Added

- Added image feature-map and Grad-CAM visualization API.
- Expanded training diagnostics with residual-vs-true-MOS and MOS-bin reports.

### Changed

- Preserved complete per-sample error manifests for downstream analysis.

## [0.7.1] - 2026-08-20

### Added

- Added RESTful FastAPI resources for health, model metadata, and media evaluations.
- Added generated Swagger UI, ReDoc, and OpenAPI documentation.
- Added feature-map and Grad-CAM visualization utilities.

### Changed

- Moved training plots into `src/visualization/`.
- Improved dataset integrity auditing for missing, corrupted, duplicate, and
  filename/label-mismatched samples.

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

[0.7.1]: https://github.com/autentisitet/deep-vqa-framework/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/autentisitet/deep-vqa-framework
[0.6.4]: https://github.com/autentisitet/deep-vqa-framework/commit/a43df487bee241484d50f7ea3e526f662ba4ba72
[0.6.2]: https://github.com/autentisitet/deep-vqa-framework/commit/dd1a191
[0.6.0]: https://github.com/autentisitet/deep-vqa-framework/commit/77fd572
[0.5.1]: https://github.com/autentisitet/deep-vqa-framework/commit/3737aec
