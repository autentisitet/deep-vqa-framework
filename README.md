# deep-vqa-framework

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub release](https://img.shields.io/github/v/release/autentisitet/deep-vqa-framework?include_prereleases)](https://github.com/autentisitet/deep-vqa-framework/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.4.3--beta-blue.svg)](https://github.com/autentisitet/deep-vqa-framework)
[![Code style: ruff](https://img.shields.io/badge/ruff-⭐-purple)](https://github.com/astral-sh/ruff)

**A Unified Deep Learning Framework for Image Quality Assessment (IQA) and Video Quality Assessment (VQA).**

This framework provides an end-to-end solution for training, evaluating, and deploying quality assessment models. It features a unified architecture that seamlessly handles both image and video inputs, multi-dataset support, cross-validation pipelines, and production-ready inference APIs.

> [!NOTE]
> This framework is primarily tested on AutoDL cloud GPU instances.

---

## Table of Contents

- [Architecture & Design Decisions](#architecture-decisions)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Metrics](#evaluation-metrics)
- [Deployment & Inference API](#deployment-api)
- [Project Main Structure](#project-main-structure)
- [System Overview](#system-overview)
- [Configuration Guide](#configuration-guide)
- [Troubleshooting](#troubleshooting)
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
| **Unified Model** | Single `IQAVQANet` handles both 4D and 5D inputs | Eliminates duplicate code, ensures consistent quality metrics |
| **Flexible Backbones** | Swin-T / ResNet50 with automatic feature adaptation | Balances accuracy vs. memory consumption |
| **Temporal Fusion** | Transformer encoder for video frame aggregation | Captures long-range dependencies between frames |
| **Task-Aware Loss** | MSE + Rank + PLCC, reweighted per `task_type` | Optimizes absolute prediction, relative ordering, and linear MOS alignment |
| **Multi-Dataset Support** | YAML-based configuration with factory pattern | Easy addition of new datasets without code changes |
| **Path Abstraction** | DSL-based `PathManager` with YAML routing | Eliminates hardcoded paths, supports symbolic links |
| **Lazy Asset Resolution** | `CaseInsensitiveAssetResolver` with pre-built index | O(1) file lookup, case-insensitive matching |

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

> [!WARNING]
> Known issue: `mode` isn't currently passed from the training engine into `IQAVQALoss.forward()`, so VQA runs fall back to the IQA weight set until this is wired up.

---

## Training Pipeline <a id="training-pipeline"></a>

### Quick Start Training

#### Step 1: Initialize Paths

```bash
make link
```

#### Step 2: Training Commands

You can choose between running the direct uv command or using the make wrapper.

| Dataset | Model | uv Command | make Command |
| :-------- | :------ | :----------- | :------------- |
| TID2013 | `resnet_iqa` | `uv run python -m src.main --model resnet_iqa --dataset tid2013` | `make train DATASET=tid2013 MODEL=resnet_iqa` |
| KoNViD-1k | `timeswin_vqa` | `uv run python -m src.main --model timeswin_vqa --dataset konvid-1k` | `make train DATASET=konvid-1k MODEL=timeswin_vqa` |
| T2VQA-DB | `timeswin_vqa` | `uv run python -m src.main --model timeswin_vqa --dataset t2vqa-db` | `make train DATASET=t2vqa-db MODEL=timeswin_vqa` |

*Note: By default, DEBUG=0 is applied in make commands. You can override it by appending DEBUG=1 if needed.*

> [!NOTE]
> Only two model configs ship today: `resnet_iqa` (image/ResNet50) and `timeswin_vqa` (video/Swin-T). Model configs are auto-discovered from `config/models/*.yaml` — drop a new YAML there (e.g. `resnet_vqa.yaml`) to register another combination before referencing it in commands.

### Configuration Parameters

```yaml
# config/models/timeswin_vqa.yaml
preprocessing:
  batch_size: 8            # Reduce if OOM
  num_workers: 4           # Data loading threads
  k_fold: 5                 # Cross-validation folds

model:
  backbone: "swin_t"        # or "resnet50"
  num_frames: 16            # Video frames per sample
  transformer_layers: 2     # Temporal fusion depth

train:
  epochs: 30
  lr: 0.00005
  gradient_accumulation_steps: 4  # Effective batch = batch_size × steps
  early_stop:
    enabled: true
    patience: 10
    monitor: "val_srocc"
    mode: "max"
```

### Advanced Options

You can extend the framework capabilities using the following training and debugging modes:

| Mode | Use Case | uv / Shell Command | make Wrapper |
| :----- | :--------- | :------------------- | :------------- |
| **Smoke Test** | Quick functionality check | `uv run python -m src.main --smoke_test` | `make test` |
| **Debug Mode** | Enable breakpoints & verbose logs | `LOG_LEVEL=DEBUG uv run python -m src.main` | `make train DEBUG=1` |
| **Background** | Run on remote server persistently | `nohup uv run python -m src.main > results/scripts_logs/train.log 2>&1 &` | `make train` |

## Detailed Execution

- **Smoke Test**: Executes a single epoch with minimal data to verify pipeline integrity. Ideal for CI/CD or validating changes.

```bash
uv run python -m src.main --smoke_test
```

- **Debug Mode**: Runs with LOG_LEVEL=DEBUG to enable verbose logging and debugger breakpoints. Useful for troubleshooting.

```bash
LOG_LEVEL=DEBUG uv run python -m src.main
```

- **Background Training**: Uses nohup to ensure training continues after terminal closure. Output is redirected to results/scripts_logs/train.log.

```bash
nohup uv run python -m src.main > results/scripts_logs/train.log 2>&1 &
```

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
> MOS denormalization uses `dataset_info.mos_min`/`mos_max` from the checkpoint's saved config if present, otherwise falls back to hardcoded constants (`DATASET_MOS_PARAMS`) for TID2013/KoNViD-1k only. Given the training pipeline doesn't currently guarantee `mos_min`/`mos_max` land in `dataset_info` (see the Configuration Guide note on `dataset_info` above), the hardcoded fallback is likely what's actually used in practice — verify the printed range in the startup log (`✅ ... 模型加载完成 (... MOS 范围: X~Y)`) matches your dataset before trusting `dataset_mos` output.

> [!NOTE]
> CORS is currently wide open (`allow_origins=["*"]`) — fine for local development, but tighten this before exposing the service beyond your own machine.

---

## Project Main Structure <a id="project-main-structure"></a>

```text
deep-vqa-framework/
├── Makefile                # Automation & workflow commands
├── README.md               # Project overview
├── DISCLAIMER.md           # Legal liability & resource usage policy
├── pyproject.toml          # Dependency & environment management (uv)
│
├── config/                 # YAML configuration modules
│   ├── paths.yaml            # Path resolution DSL (roots & resolvers)
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
│
├── results/                  # Global outputs & logs
│   ├── model_outputs/           # Training checkpoints
│   ├── train_logs/               # Execution & performance history
│   └── plots/                     # Visualization (loss, residuals, etc.)
│
├── scripts/                  # Infrastructure automation
│   ├── manage_data.sh           # Download & data preparation
│   ├── setup_env.sh              # Environment & system initialization
│   ├── archive_results.sh         # Package results
│   └── *.sh                        # Auxiliary maintenance & cleanup scripts
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
    └── utils/                        # Configuration, logging & path management
```

---

## System Overview <a id="system-overview"></a>

For a detailed look at the system architecture and execution flow, we provide two viewing options:

- [**Interactive Architecture Map**](docs/pipeline.html)
- [**Static Architecture Overview**](docs/pipeline.png)

---

## Configuration Guide <a id="configuration-guide"></a>

### Configuration Layering

Configuration is assembled by `config_loader.load_system_config()` in two distinct steps:

| Stage | File | How it's merged |
| ------- | ------ | --------- |
| 1 (Base) | `basic.yaml` | Loaded first as the starting config |
| 2 (Model) | `models/{model}.yaml` | Deep-merged on top of stage 1 (matching keys override) |
| 3 (Dataset) | `dataset_config.yaml` | **Not merged into top-level keys** — the matched dataset entry is attached wholesale as `config["dataset_info"]` |

The merged result is validated against `config_loader.validate_config_schema()`'s required-field list before training starts. Path resolution (`config/paths.yaml`) is handled separately by `PathManager` and is not part of this merge.

### Memory Optimization for Video Training

```yaml
# If encountering CUDA Out of Memory (OOM)
preprocessing:
  batch_size: 1              # Reduce batch size
  num_workers: 0             # Disable multiprocessing

model:
  num_frames: 4              # Reduce temporal frames
  backbone: "resnet50"       # Use smaller backbone
  transformer_layers: 1      # Reduce transformer depth

train:
  gradient_accumulation_steps: 4  # Simulate larger batch
  amp: true                  # Enable mixed precision
```

---

## Troubleshooting <a id="troubleshooting"></a>

### CUDA Out of Memory

| Symptom | Solution |
| :--- | :--- |
| OOM at first batch | Reduce `batch_size` to 1 |
| OOM after several epochs | Reduce `num_frames` or switch to the `resnet50` backbone |
| OOM during validation | Reduce `num_frames` to 4 |

> [!NOTE]
> Gradient checkpointing is not currently implemented in `IQAVQANet` — don't set `gradient_checkpointing: true` in configs yet; it has no effect.

### Dataset Not Found

If you encounter `FileNotFoundError` when passing `--dataset xxx`, it means the dataset symlink is missing or incorrect.

```bash
make link
```

or you can:

```bash
cd scripts
bash setup_links.sh
```

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

`quarantine/` (created by `DataEDA.check_integrity()` when corrupted files are moved aside) is **not** currently handled by `cache_clean.sh` — clear it manually, or add a cleanup step to the script.

### Training Hangs With No Error (Background Runs)

If a `nohup`/background training run appears frozen with no new log lines and no crash, check whether it's stuck at a `pdb.set_trace()` breakpoint. `PathManager.resolve(..., mkdir=True)` currently drops into `pdb` on `PermissionError`/`OSError` while creating a directory, which will silently block a non-interactive process waiting on stdin instead of raising. Kill the process and check disk permissions/space if this happens.

---

## 📄 License <a id="license"></a>

- **Framework**: [MIT](LICENSE)
- **Author**: [@autentisitet](https://github.com/autentisitet)
- **Version**: 0.4.3-beta (pre-release)

---

## 👥 Contributors <a id="contributors"></a>

| Name | Role | Contributions |
| :--- | :--- | :--- |
| **[@autentisitet](https://github.com/autentisitet)** | Project Lead / Core Developer | Framework architecture, training pipeline, inference engine, deployment API |
| **[@yss0120](https://github.com/yss0120)** | Frontend Developer | Interactive frontend UI/UX (`index.html`), subjective blind rating system, quality passport visualization, data EDA pipeline |
| **[@Zed-23](https://github.com/Zed-23)** | DevOps & QA Engineer | GitHub Actions CI configuration, shell script fixes, smoke testing, CUDA OOM debugging and memory optimization |
| **[@bazhina-5566](https://github.com/bazhina-5566)** | Backend API Developer | FastAPI service (`deploy/api.py`), model checkpoint integration, inference API design |

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
