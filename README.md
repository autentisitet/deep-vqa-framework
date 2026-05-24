# deep-vqa-framework

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub release](https://img.shields.io/github/v/release/autentisitet/deep-vqa-framework?include_prereleases)](https://github.com/autentisitet/deep-vqa-framework/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-blue)](https://github.com/autentisitet/deep-vqa-framework)
[![Version](https://img.shields.io/badge/version-0.9.2--beta-blue.svg)](https://github.com/autentisitet/deep-vqa-framework)

**A Unified Deep Learning Framework for Image Quality Assessment (IQA) and Video Quality Assessment (VQA).**

This framework provides an end-to-end solution for training, evaluating, and deploying quality assessment models. It features a unified architecture that seamlessly handles both image and video inputs, multi-dataset support, cross-validation pipelines, and production-ready inference APIs.

> [!NOTE]
> This framework is primarily tested on AutoDL cloud GPU instances.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Architecture & Design Decisions](#architecture-decisions)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Metrics](#evaluation-metrics)
- [Project Main Structure](#project-main-structure)
- [System Overview](#system-overview)
- [Configuration Guide](#configuration-guide)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgments)

---

## System Requirements <a id="system-requirements"></a>

### Hardware Requirements

| Component | Minimum | Recommended |
| ----------- | --------- | ------------- |
| **GPU** | 8GB VRAM (IQA only) | 24GB+ VRAM (VQA training) |
| **RAM** | 16GB | 32GB+ |
| **Disk Space** | 50GB | 200GB+ (including datasets) |
| **CPU** | 4 cores | 8+ cores |

### Software Requirements

| Component | Version | Notes |
| ----------- | --------- | ------- |
| **OS** | Linux (Ubuntu 20.04+) / Windows 10+ / macOS 12+ | Linux recommended for training |
| **Python** | 3.10 - 3.12 | 3.12+ not fully tested |
| **CUDA** | 11.8 / 12.1 | Required for GPU training |
| **PyTorch** | 2.0+ | 2.5+ recommended for better AMP support |
| **cuDNN** | 8.7+ | Included with PyTorch |

### Storage Breakdown

The framework expects datasets in the following structure:

| Dataset | Size (Compressed) | Size (Extracted) |
| --------- | ------------------- | ------------------- |
| TID2013 | ~500MB | ~3GB |
| KoNViD-1k | ~9GB | ~10GB |
| T2VQA-DB | ~45GB | ~50GB+ |
| **Total** | **~55GB** | **~63GB+** |

### Additional Space

| Item | Estimated Size |
| ------ | ---------------- |
| Python environment (uv/venv) | ~5GB |
| Model checkpoints (5-fold) | ~10GB |
| Training logs & plots | ~2GB |
| **Grand Total** | **~80-100GB** |

> [!NOTE]
> - Use `uv` for faster installation and smaller dependency footprint
> - Symbolic links (see below) do not consume additional disk space
> - `quarantine/` directory may grow if files are isolated; run `scripts/cache_clean.sh` regularly

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
| **Hybrid Loss** | MSE (70%) + Rank Loss (30%) | Optimizes both absolute prediction and relative ordering |
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

### Loss Function: Hybrid MSE + Rank Loss

```text
Total Loss = 0.7 × MSE + 0.3 × Rank Loss

- MSE Loss: Absolute prediction accuracy
- Rank Loss: Preserves relative ordering between samples
```

---

## Training Pipeline <a id="training-pipeline"></a>

### Quick Start Training

#### Step 1: Create symbolic links

```bash
make link
```

or you can:

```bash
cd scripts
bash setup_links.sh
```

#### Step 2: Run training

- **Image Dataset (TID2013)**

```bash
uv run python -m src.main --model resnet_iqa --dataset tid2013
```

or you can:

```bash
make train DATASET=tid2013 MODEL=resnet_iqa DEBUG=0
```

- **Video Dataset (KoNViD-1k)**

```bash
uv run python -m src.main --model timeswin_vqa --dataset konvid-1k
```

or you can:

```bash
make train DATASET=konvid-1k MODEL=timeswin_vqa DEBUG=0
```

- **Video Dataset (T2VQA-DB)**

```bash
uv run python -m src.main --model resnet_vqa --dataset t2vqa-db
```

or you can:

```bash
make train DATASET=t2vqa-db MODEL=resnet_vqa DEBUG=0
```

### Configuration Parameters

```yaml
# config/models/resnet_vqa.yaml
preprocessing:
  batch_size: 2          # Reduce if OOM
  num_workers: 4         # Data loading threads
  k_fold: 5              # Cross-validation folds

model:
  backbone: "resnet50"   # or "swin_t"
  num_frames: 8          # Video frames per sample
  transformer_layers: 2  # Temporal fusion depth

train:
  epochs: 50
  lr: 0.0001
  gradient_accumulation_steps: 4  # Effective batch = batch_size × steps
  early_stop:
    enabled: true
    patience: 10
    monitor: "val_srocc"
    mode: "max"
```

### Advanced Options

- **Fast Smoke Test**

```bash
uv run python -m src.main --model resnet_iqa --dataset tid2013 --smoke_test
```

or you can:

```bash
make test DATASET=tid2013 MODEL=resnet_iqa
```

- **Debug Mode (with breakpoints)**

```bash
LOG_LEVEL=DEBUG uv run python -m src.main --model resnet_iqa --dataset tid2013
```

or you can:

```bash
make train DATASET=tid2013 MODEL=resnet_iqa DEBUG=1
```

- **Background Training**

```bash
nohup python -m src.main --model resnet_vqa --dataset t2vqa-db > results/scripts_logs/train.log 2>&1 &
tail -f results/scripts_logs/train.log
```

or you can:

```bash
make train DATASET=t2vqa-db MODEL=resnet_vqa DEBUG=1
```

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

### Output Example

```text
📊 [Epoch 050 | HYPERIQANET] PLCC: 0.9696 | SROCC: 0.9689 | RMSE: 0.0442 | R²: 0.9314
```

### Visualizations

The framework automatically generates:

- **Training History**: Loss curves, PLCC/SROCC progression

- **Residual Analysis**: Scatter plots, error distribution

- **Cross-Model Comparison**: Bar charts for multiple models

Output location: `results/{dataset}/plots/`

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
│   ├── basic.yaml          # System & training global defaults
│   ├── dataset_config.yaml # Dataset-specific metadata
│   └── models/             # Model architecture parameters
│
├── datasets/               # Data storage & symlink routing
│   ├── KoNViD-1k/          # Video quality dataset
│   ├── T2VQA-DB/           # Text-to-Video QA dataset
│   └── TID2013/            # Image quality dataset
│
├── docs/                   # Interactive architecture & manuals
│   ├── pipeline.html       # System execution & module flow
│   └── Cloud_Platform_Rental_Guide.md
│
├── results/                # Global outputs & logs
│   ├── model_outputs/      # Training checkpoints
│   ├── train_logs/         # Execution & performance history
│   └── plots/              # Visualization (loss, residuals, etc.)
│
├── scripts/                # Infrastructure automation
│   ├── manage_data.sh      # Download & data preparation
│   ├── setup_env.sh        # Environment & system initialization
│   └── *.sh                # Auxiliary maintenance & cleanup scripts
│
└── src/                    # Core framework logic
    ├── main.py             # Global execution entry point
    ├── core/               # Training engine & evaluation pipeline
    ├── data/               # Data loaders, EDA & integrity analysis
    ├── models/             # Architecture definitions (IQAVQA-Net)
    └── utils/              # Configuration, logging & path management
```

---

## System Overview <a id="system-overview"></a>

For a detailed look at the system architecture and execution flow, we provide two viewing options:

- [**Interactive Architecture Map**](docs/pipeline.html)
- [**Static Architecture Overview**](docs/pipeline.png)

---

## Configuration Guide <a id="configuration-guide"></a>

### Configuration Layering

Configuration files are merged in the following order (later files override earlier ones):

| Layer | File | Purpose |
| ------- | ------ | --------- |
| 1 (Base) | `basic.yaml` | Global defaults |
| 2 (Model) | `models/{model}.yaml` | Model-specific overrides |
| 3 (Dataset) | `dataset_config.yaml` | Dataset-specific settings |

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
| OOM after several epochs | Enable `gradient_checkpointing: true` |
| OOM during validation | Reduce `num_frames` to 4 |

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

---

## 📄 License <a id="license"></a>

- **Framework**: [MIT](LICENSE)
- **Author**: [@autentisitet](https://github.com/autentisitet)
- **Version**: 0.9.2-beta (pre-release)

---

## 🙏 Acknowledgments <a id="acknowledgments"></a>

- PyTorch team for deep learning framework
- Decord developers for efficient video loading
- TID2013, KoNViD-1k, T2VQA-DB dataset providers

---

## ⚖️ Legal & Disclaimer
For details regarding third-party tool usage, dataset compliance, and resource usage, please refer to the [DISCLAIMER.md](DISCLAIMER.md) file.

---

**Built with ❤️ for the research community**
