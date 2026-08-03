# deep-vqa-framework

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub release](https://img.shields.io/github/v/release/autentisitet/deep-vqa-framework?include_prereleases)](https://github.com/autentisitet/deep-vqa-framework/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.6.0--beta-blue.svg)](https://github.com/autentisitet/deep-vqa-framework)
[![Code Quality: ruff+black+isort+mypy](https://img.shields.io/badge/code%20quality-ruff%2Bblack%2Bisort%2Bmypy-4B8BBE.svg)](https://github.com/autentisitet/deep-vqa-framework)
[![Security: pip-audit+sbom](https://img.shields.io/badge/security-pip--audit%2Bsbom-9cf.svg)](https://github.com/autentisitet/deep-vqa-framework)

**🌐 [English](README.md) | [简体中文](README_zh.md)**

**A Unified Deep Learning Framework for Image Quality Assessment (IQA) and Video Quality Assessment (VQA).**

This framework provides an end-to-end solution for training, evaluating, and deploying quality assessment models. It features a unified architecture that seamlessly handles both image and video inputs, multi-dataset support, cross-validation pipelines, and production-ready inference APIs.

> [!NOTE]
> This framework supports both Docker and Podman container runtimes.
> For dataset downloads, the scripts automatically detect `http_proxy`/`HTTP_PROXY` environment variables.
>
> On AutoDL cloud GPU instances, you can enable proxy with:
>
```bash
source /etc/network_turbo
```

> [!TIP]
> **For Podman users**: No alias is required for `make docker-*` commands.
> The Makefile auto-detects your runtime and uses `podman-compose` or `docker-compose` accordingly.
>
> However, if you want to run `docker` commands manually, you can set an alias:
>
```bash
alias docker=podman
alias docker-compose=podman-compose
```

---

## Table of Contents

- [Architecture & Design Decisions](#architecture-decisions)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Metrics](#evaluation-metrics)
- [Deployment & Inference API](#deployment-api)
- [Project Main Structure](#project-main-structure)
- [Docker / Podman Support](#docker-support)
- [System Overview](#system-overview)
- [Configuration Guide](#configuration-guide)
- [Troubleshooting](#troubleshooting)
- [Dependency Security](#dependency-security)
- [License](#license)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgments)

---

## Architecture & Design Decisions <a id="architecture-decisions"></a>

### Unified IQA/VQA Architecture

The framework implements a dimension-aware routing system that automatically switches between image (4D tensors) and video (5D tensors) processing modes.

**Key Design Decisions:**

| Decision | Implementation | Rationale |
| :--- | :--- | :--- |
| **Unified Model** | A single `IQAVQANet` handling both 4D and 5D inputs | Eliminates code duplication; ensures consistency in quality metrics |
| **Flexible Backbone** | Swin-T / ResNet50 with automatic feature adaptation | Balances accuracy against GPU memory consumption |
| **Temporal Fusion** | Transformer encoder for video frame aggregation | Captures long-range dependencies between frames |
| **Task-Aware Loss** | MSE + Rank + PLCC, reweighted based on `task_type` | Simultaneously optimizes absolute prediction and relative ranking |
| **Multi-dataset Support** | YAML-based configuration and Factory pattern | Allows easy addition of new datasets without code modification |
| **Config & Path Management** | Pydantic-based `Config.paths` and typed methods | Single source of truth; eliminates hard-coding; enables dataset-isolated storage |
| **File Index Resolution** | `CaseInsensitiveAssetResolver` pre-builds a case-insensitive index | Reduces O(n) directory traversal to O(1) memory lookup; resolves parsing failures caused by filename case inconsistencies |
| **Model Serving** | FastAPI + secure loading via `weights_only=True` | Decouples training from inference; provides standardized HTTP interfaces; ensures secure model loading |

---

## Model Architecture <a id="model-architecture"></a>

### IQAVQANet: Unified Quality Assessment Network

```python
# Architecture overview
Input (4D: [B,3,H,W] or 5D: [B,F,3,H,W])
    ↓
Backbone (Swin-T / ResNet50)
    ↓
Spatial Pooling (AdaptiveAvgPool2d)
    ↓
[Temporal Fusion] ← TransformerEncoder (only for video)
    ↓
Quality Head (3-layer MLP + Sigmoid)
    ↓
Output: Quality Score (0-1 range)
```

### Supported Configurations

| Backbone | Parameters | IQA | VQA | Memory (per sample) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | 25M | ✅ | ✅ | ~2GB (8 frames) |
| **Swin-T** | 28M | ✅ | ✅ | ~4GB (8 frames) |

### Loss Function: Task-Aware Hybrid Loss

`IQAVQALoss` combines three components, weighted differently depending on `task_type` (`iqa` vs `vqa`):

```text
Total Loss = w_mse × MSE + w_rank × RankLoss + w_plcc × (1 − PLCC)
```

| Task | MSE | Rank | PLCC |
| :--- | :-- | :--- | :--- |
| IQA (`resnet_iqa`) | 0.7 | 0.3 | 0.0 |
| VQA (`timeswin_vqa`) | 0.4 | 0.3 | 0.3 |

- **MSE Loss**: Absolute prediction accuracy
- **Rank Loss**: Pairwise ranking consistency (sampled, capped at `max_pairs` pairs)
- **PLCC Loss**: `1 − Pearson correlation`, weighted in for VQA to directly optimize linear alignment with human MOS

---

## Training Pipeline <a id="training-pipeline"></a>

### Quick Start Training

#### Step 1: Environment Setup

```bash
# Initialize environment and install dependencies
make install

# Check environment status
make info

# Download datasets, unrar and make symbol links
make data
```

#### Step 2: Training Commands

Run training directly with uv:

```bash
# TID2013 (Image Quality Assessment)
uv run python -m src.main --model resnet_iqa --dataset tid2013

# KoNViD-1k (Video Quality Assessment)
uv run python -m src.main --model timeswin_vqa --dataset konvid-1k

# T2VQA-DB (Text-to-Video Quality Assessment)
uv run python -m src.main --model timeswin_vqa --dataset t2vqa-db
```

*Note: By default, DEBUG=0 is applied in make commands. You can override it by appending DEBUG=1 if needed.*

> [!NOTE]
> Only two model configs ship today: `resnet_iqa` (image/ResNet50) and `timeswin_vqa` (video/Swin-T). Model configs are auto-discovered from `config/models/*.yaml` — drop a new YAML there (e.g. `resnet_vqa.yaml`) to register another combination before referencing it in commands.

### Advanced Options

You can extend the framework capabilities using the following training and debugging modes:

| Mode | Use Case | uv / Shell Command |
| :----- | :--------- | :------------------- |
| **Smoke Test** | Quick functionality check | `uv run python -m src.main --smoke_test` |
| **Debug Mode** | Enable breakpoints & verbose logs | `LOG_LEVEL=DEBUG uv run python -m src.main` |
| **Background** | Run on remote server persistently | `nohup uv run python -m src.main > results/scripts_logs/train.log 2>&1 &` |

> [!TIP]
> Monitor real-time training progress with:
>
> ```bash
> tail -f results/scripts_logs/train.log
> ```

---

## Evaluation & Metrics <a id="evaluation-metrics"></a>

### Core Metrics

| Metric | Full Name | Interpretation |
| :--- | :--- | :--- |
| **PLCC** | Pearson Linear Correlation Coefficient | Linear relationship (accuracy) |
| **SROCC** | Spearman Rank Order Correlation Coefficient | Monotonic relationship (ranking) |
| **KROCC** | Kendall Rank Correlation Coefficient | Ordinal agreement |
| **RMSE** | Root Mean Square Error | Prediction error magnitude |
| **R²** | Coefficient of Determination | Variance explained |

### Visualizations

The framework automatically generates:

- **Training History**: Loss curves, PLCC/SROCC progression

- **Residual Analysis**: Scatter plots, error distribution

- **Cross-Model Comparison**: Bar charts for multiple models

Output location: `results/{dataset}/plots/`

---

## Deployment & Inference API <a id="deployment-api"></a>

A standalone FastAPI service (`deploy/api.py`) exposes trained checkpoints for inference, decoupled from the training stack. It loads one IQA model and one VQA model at startup and serves a unified 0-5 MOS scale regardless of which model answers the request.

### Directory Layout

```text
deploy/
├── api.py               # FastAPI service (this file)
├── infer.py              # Preprocessing + checkpoint loading + prediction helpers
├── iqa-models/
│   └── tid2013_best.pt   # Default IQA checkpoint (resnet_iqa, trained on TID2013)
└── vqa-models/
    └── konvid_best.pt    # Default VQA checkpoint (timeswin_vqa, trained on KoNViD-1k)
```

> [!WARNING]
> Checkpoint paths are resolved relative to `api.py`'s own location (`Path(__file__).resolve().parent / "iqa-models" / ...`), so `.pt` files must sit in `deploy/iqa-models/` and `deploy/vqa-models/` — not in `results/model_outputs/`.

### Starting the Service

```bash
cd deploy
uv run python -m deploy.api
```

The service listens on `0.0.0.0:8000` and loads both checkpoints eagerly on startup; if neither model file is found, startup fails with `RuntimeError: 没有成功加载任何模型，服务启动失败`.

### Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | GET | Returns loaded model IDs and inference device |
| `/evaluate` | POST | Runs inference on an uploaded image/video and returns a unified MOS score |

`/evaluate` accepts `multipart/form-data` with:

| Field | Type | Notes |
| :--- | :--- | :--- |
| `file` | file | Image or video to score |
| `media_type` | string | `"image"` or `"video"` |
| `task_type` | string | Echoed back in the response, not used for routing |
| `model` | string | `"resnet_iqa"` or `"timeswin_vqa"` — selects which cached model handles the request |

Cross-media inference is supported in both directions: `resnet_iqa` averages predictions across sampled frames when given a video, and `timeswin_vqa` expands a single image into a pseudo-video (`image_to_video_tensor`) when given an image.

The response includes both `mos` (unified 0–5 scale, `raw_score × 5`) and `dataset_mos` (denormalized back to the source dataset's original MOS range, for debugging).

> [!NOTE]
> MOS range (`mos_min`/`mos_max`) is stored in the checkpoint's config and restored on load for checkpoints saved with v0.5.0+. Denormalization is automatic.

> [!TIP]
> The `dataset_mos` field returns scores in the original dataset's MOS scale (e.g., TID2013: 0-9, KoNViD-1k: 1-5), while `mos` is always normalized to 0-5 for cross-dataset comparison.

---

## Project Main Structure <a id="project-main-structure"></a>

```text
deep-vqa-framework/
├── Makefile                # Automation & workflow commands
├── README.md               # Project overview
├── DISCLAIMER.md           # Legal liability & resource usage policy
├── pyproject.toml          # Dependency & environment management (uv)
│
├── config/                 # YAML configuration files (user-editable)
│   ├── basic.yaml            # System & training global defaults
│   ├── dataset_config.yaml   # Dataset-specific metadata
│   └── models/                 # Model architecture parameters
│
├── datasets/                 # Data storage & symlink routing
│   ├── KoNViD-1k/               # Video quality dataset
│   ├── T2VQA-DB/                 # Text-to-Video QA dataset
│   └── TID2013/                  # Image quality dataset
│
├── docs/                     # Interactive architecture & manuals
│   ├── pipeline.html            # System execution & module flow
│   └── Cloud_Platform_Rental_Guide.md
|
├── reports/                  # Security reports from pip-audit, SBOM, and safety
|
├── results/
|   ├── {dataset}/
│   │   ├── train_logs/           # Training history, CSV logs
│   │   ├── plots/                # Loss curves, residual plots
│   │   ├── model_outputs/        # Checkpoints (.pt files)
│   │   └── corrupted/            # Corrupted files from integrity check
│   └── scripts_logs/             # Shell script logs (setup, data, etc.)
|
├── docker/                   # Container configuration
│   ├── docker-compose.yaml      # Main compose config
│   ├── docker-compose.docker.yaml # Docker GPU support
│   └── docker-compose.podman.yaml # Podman GPU support
│
├── .github/workflows/        # CI/CD pipelines
│   └── ci.yaml                 # Continuous Integration
|
├── scripts/                  # Infrastructure automation
│   ├── bootstrap.sh             # System-level initialization (apt, mirrors, system tools)
│   ├── setup_env.sh             # Project-level initialization (uv, .venv, Python deps)
│   ├── manage_data.sh           # Download & data preparation
│   ├── archive_results.sh       # Package results
│   ├── cache_clean.sh           # Cache cleanup
│   └── ci_test_extract.sh       # CI helper for smart_extract test
│
├── deploy/                  # Standalone inference service (decoupled from training)
│   ├── api.py                    # FastAPI service — see note below on module naming
│   ├── infer.py                   # Preprocessing + checkpoint loading + prediction
│   ├── iqa-models/                # IQA .pt checkpoints served by api.py
│   └── vqa-models/                # VQA .pt checkpoints served by api.py
│
└── src/                       # Core framework logic
    ├── main.py                   # Global execution entry point
    ├── core/                        # Training engine & evaluation pipeline
    ├── data/                        # Data loaders, EDA & integrity analysis
    ├── models/                      # Architecture definitions (IQAVQA-Net)
    ├── utils/                        # Configuration, logging & path management
    └── config/                     # Pydantic config system (code)
```

---

## Docker / Podman Support <a id="docker-support"></a>

The framework supports containerized development and deployment with both Docker and Podman.

### Quick Start with Docker

```bash
# Build and enter development container
make docker-dev

# Run training in container
make docker-train

# Start inference API service
make docker-infer

# Stop all containers
make docker-stop

# Check container environment
make docker-manage
```

### Container Configuration

| Component | Description |
| :--- | :--- |
| `Dockerfile` | Multi-stage builds: `base` (shared deps), `train` (training), `prod` (inference) |
| `docker-compose.yaml` | Main compose configuration |
| `docker-compose.docker.yaml` | Docker-specific GPU support (`runtime: nvidia` + `environment` ) |
| `docker-compose.podman.yaml` | Podman-specific GPU support (`security_opt` + `devices`) |

---

## System Overview <a id="system-overview"></a>

To help you quickly grasp the system architecture and execution flow, we provide an **interactive pipeline visualization**.

[**→ Open Interactive Architecture Map**](docs/pipeline.html)

This map illustrates:
- How data flows through each stage of the system
- Key components and their interactions
- Execution order of the entire workflow

> Charts are rendered using Mermaid and support interactive viewing in the browser.

---

## Configuration Guide <a id="configuration-guide"></a>

Configuration is assembled by `load_config()` which returns a Pydantic `Config` object.
All settings are validated at load time with type checking.
Paths are resolved via `cfg.paths.xxx_dir(dataset_name)` methods.

| Stage | File | How it's merged |
| ------- | ------ | --------- |
| 1 (Base) | `basic.yaml` | Loaded first as the starting config |
| 2 (Model) | `models/{model}.yaml` | Deep-merged on top of stage 1 (matching keys override) |
| 3 (Dataset) | `dataset_config.yaml` | **Not merged into top-level keys** — the matched dataset entry is attached wholesale as `config["dataset_info"]` |

The merged result is validated against `config_loader.validate_config_schema()`'s required-field list before training starts. Path resolution is handled by Pydantic `PathsConfig` with typed methods.

---

## Troubleshooting <a id="troubleshooting"></a>

### CUDA Out of Memory (OOM)

When OOM occurs, adjust the following parameters in your model's YAML config (e.g., `config/models/timeswin_vqa.yaml`):

**Reduce memory consumption:**

- Lower `preprocessing.batch_size` — reduces per-batch memory
- Reduce `model.num_frames` — fewer video frames to process
- Switch to a lighter `model.backbone` (e.g., `resnet50` instead of `swin_t`)
- Lower `model.transformer_layers` — shallower temporal fusion

**Compensate for smaller batch size:**

- Increase `train.gradient_accumulation_steps` — maintains effective batch size = `batch_size × gradient_accumulation_steps`
- Ensure `amp: true` — mixed precision significantly reduces memory

> [!NOTE]
> Gradient checkpointing is not currently implemented in `IQAVQANet` — don't set `gradient_checkpointing: true` in configs yet; it has no effect.

### Video Loading Backend (AutoDL Specific)

> [!WARNING]
On AutoDL or similar cloud GPU instances, OpenCV's VideoCapture may fail due to missing system dependencies.

Solution: Use Decord

```bash
uv add decord
```

Decord is pre-configured as the default backend. If Decord is not available, the framework automatically falls back to OpenCV, but on AutoDL this fallback may fail. Always use Decord for video training on AutoDL.

### Slow Training

| Issue | Optimization |
| :--- | :--- |
| Data loading bottleneck | Increase `num_workers: 8` |
| Small batch size | Use `gradient_accumulation_steps` |
| Video decoding slow | Ensure Decord is installed |

### Disk Filling Up

`results/{dataset}/corrupted/` stores files moved aside by `DataEDA.check_integrity()`.
- Clear manually if no longer needed
- Or add a cleanup step to `scripts/cache_clean.sh`

> [!TIP]
> These files are corrupted media files isolated during integrity checks — safe to delete if you don't need them for debugging.

### Training Hangs With No Error (Background Runs)

If a `nohup`/background training run appears frozen with no new log lines, check for `pdb.set_trace()` breakpoints in `main.py` or `engine.py`. These are triggered on exceptions and will wait for stdin input, blocking non-interactive processes.

**Solutions:**
- Run training interactively (without `nohup`) when debugging
- Check `results/{dataset}/train_logs/*.log` for error details
- Kill the process and fix the underlying issue before retrying

---

## Dependency Security <a id="dependency-security"></a>

The framework includes security tools to audit dependencies:

| Command | Purpose |
| :--- | :--- |
| `make vuln-audit` | Scan dependencies for known vulnerabilities |
| `make sbom` | Generate Software Bill of Materials (CycloneDX) |
| `make safety` | Check dependencies with Safety (legacy, requires login) |
| `make security-all` | Run all security checks |

> [!NOTE]
> `pip-audit` is the primary vulnerability scanner. `safety` requires registration or login.

---

## 📄 License <a id="license"></a>

- **Framework**: [MIT](LICENSE)
- **Author**: [@autentisitet](https://github.com/autentisitet)
- **Version**: 0.6.0-beta (pre-release)

---

## 👥 Contributors <a id="contributors"></a>

| Name | Role | Contributions |
| :--- | :--- | :--- |
| **[@autentisitet](https://github.com/autentisitet)** | Project Lead / Core Developer | Framework architecture, training pipeline, inference engine,  Docker/Podman containerization, multi-stage builds, inference API design |
| **[@yss0120](https://github.com/yss0120)** | Frontend Developer | Interactive UI/UX (`index.html`), subjective blind rating system, quality passport visualization |
| **[@Zed-23](https://github.com/Zed-23)** | DevOps & Quality Assurance | CI/CD pipeline, automated & smoke testing, Shell script fixes, CUDA OOM debugging |
| **[@bazhina-5566](https://github.com/bazhina-5566)** | Backend API Developer | FastAPI service (`deploy/api.py`), model checkpoint loading, deploy API design |

> [!NOTE]
> We welcome contributions! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

---

## 🙏 Acknowledgments <a id="acknowledgments"></a>

- PyTorch team for deep learning framework
- Decord developers for efficient video loading
- FastAPI for the production-ready API framework
- TID2013, KoNViD-1k, T2VQA-DB dataset providers

---

## ⚖️ Legal & Disclaimer
For details regarding third-party tool usage, dataset compliance, and resource usage, please refer to the [DISCLAIMER.md](DISCLAIMER.md) file.

---

For detailed contribution guidelines and issue reporting, please check the .github folder.

**Built with ❤️ for the research community**
