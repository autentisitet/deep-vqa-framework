# Changelog

All notable changes to Deep-VQA-Framework are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/).

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

[0.7.0]: https://github.com/autentisitet/deep-vqa-framework
[0.6.4]: https://github.com/autentisitet/deep-vqa-framework/commit/a43df487bee241484d50f7ea3e526f662ba4ba72
[0.6.2]: https://github.com/autentisitet/deep-vqa-framework/commit/dd1a191
[0.6.0]: https://github.com/autentisitet/deep-vqa-framework/commit/77fd572
[0.5.1]: https://github.com/autentisitet/deep-vqa-framework/commit/3737aec
