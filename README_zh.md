# deep-vqa-framework

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-red.svg)](https://pytorch.org/)
[![GitHub release](https://img.shields.io/github/v/release/autentisitet/deep-vqa-framework?include_prereleases)](https://github.com/autentisitet/deep-vqa-framework/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.6.4-blue.svg)](https://github.com/autentisitet/deep-vqa-framework)
[![Code Quality: ruff+black+isort+mypy](https://img.shields.io/badge/code%20quality-ruff%2Bblack%2Bisort%2Bmypy-4B8BBE.svg)](https://github.com/autentisitet/deep-vqa-framework)
[![Security: pip-audit+sbom](https://img.shields.io/badge/security-pip--audit%2Bsbom-9cf.svg)](https://github.com/autentisitet/deep-vqa-framework)

**🌐 [English](README.md) | [简体中文](README_zh.md)**

**一个从研究训练走到推理部署的图像/视频质量评估框架。**

Deep-VQA-Framework 是面向生产交付的 IQA/VQA 质量评估框架，覆盖数据审计、训练评估、结果可视化、checkpoint 发布和推理服务化。
它在数据侧先隔离坏样本并备份标签，再按参考内容分组避免失真版本跨 split 泄漏；在模型侧根据 4D/5D 张量自动选择 ResNet50 或 Swin-T 的 ImageNet 特征路径，训练完成后产出可直接进入 `deploy/` 推理链路的 `{dataset}_best.pt`。

### 核心亮点

| 需求 | 内置处理 |
| :--- | :--- |
| IQA/VQA 共用入口 | 根据张量维度路由：4D 图像走 ResNet50，5D 视频走 Swin-T |
| 图像/视频输入可比 | 两条分支统一 RGB、224x224、`[0, 1]`、ImageNet mean/std 预处理 |
| 防止评估泄漏 | train/val/test 和 K 折都使用分组划分，同一参考内容只会出现在一个 split |
| 坏数据隔离 | 缺失/损坏样本在划分前剔除，对应标签备份到 `results/{dataset}/corrupted/labels/` |
| 不只保存日志 | 自动生成 EDA 图、fold 训练曲线、残差图、对比图和 fold summary |
| 部署交付闭环 | 成功训练后复制最佳 checkpoint 到 `deploy/iqa-models/{dataset}_best.pt` 或 `deploy/vqa-models/{dataset}_best.pt` |
| 容器重复运行友好 | Docker/Podman 挂载 `.cache`，复用 uv 缓存和 torchvision 预训练权重 |

```text
metadata + media
  -> 完整性检查与坏数据隔离
  -> EDA + MOS 归一化
  -> 分组 train/val/test 划分
  -> 解码时应用 ImageNet 预处理的分组 K 折训练
  -> 训练图表 + 最佳 checkpoint
  -> deploy/{iqa-models|vqa-models}/{dataset}_best.pt
```

---

## 目录

- [架构与设计决策](#architecture-decisions)
- [模型架构](#model-architecture)
- [训练流程](#training-pipeline)
- [评估与指标](#evaluation-metrics)
- [部署与推理 API](#deployment-api)
- [项目主要结构](#project-main-structure)
- [Docker / Podman 支持](#docker-support)
- [系统概览](#system-overview)
- [配置指南](#configuration-guide)
- [故障排查](#troubleshooting)
- [依赖项安全](#dependency-security)
- [许可证](#license)
- [贡献者](#contributors)
- [致谢](#acknowledgments)

---

## 架构与设计决策 <a id="architecture-decisions"></a>

### 设计约束

这个项目把 IQA/VQA 训练流程拆成明确约束，而不是松散脚本串联。

| 约束 | 实现方式 |
| :--- | :--- |
| **模态路由** | 4D 张量视为图像 batch，5D 张量视为视频 batch；通道布局不合法会提前失败 |
| **Backbone 策略** | 图像使用 ResNet50 ImageNet 特征和 BN；视频使用 Swin-T ImageNet 特征、LN 与时序融合 |
| **预处理** | 图像和采样视频帧共享 RGB、224x224、`[0, 1]`、ImageNet mean/std 归一化 |
| **数据剔除** | 缺失/损坏媒体在 EDA 和划分前剔除；标签备份，不会改写成 0 分 |
| **划分隔离** | train/val/test 和 K 折都做分组划分，同一参考内容只进入一个分区 |
| **产物路径** | 输出使用小写数据集 key 放在 `results/{dataset}/`；成功训练后发布 `{dataset}_best.pt` 到 `deploy/` |
| **运行配置** | 训练和推理共用 Pydantic 类型化配置，不再散落硬编码路径 |

---

## 模型架构 <a id="model-architecture"></a>

### IQAVQANet：统一质量评估网络

```text
# 数据预处理 + IQAVQANet / ImageNet 预训练 backbone 适配

图像媒体
  -> RGB 解码
  -> resize 到 224x224
  -> 像素缩放到 [0, 1]
  -> ImageNet mean/std 归一化
  -> 模型输入 [B, 3, 224, 224]，按 4D 路由到 IQA 分支
  -> ResNet50(IMAGENET1K_V1): conv1 + bn1 + layer1..4
  -> AdaptiveAvgPool2d
  -> 图像特征 [B, 2048]
  -> 质量预测头
  -> 质量分数 [B]

视频媒体
  -> 帧采样/补齐到 num_frames
  -> 逐帧 RGB 解码
  -> resize 到 224x224
  -> 像素缩放到 [0, 1]
  -> ImageNet mean/std 归一化
  -> 模型输入 [B, F, 3, 224, 224]，按 5D 路由到 VQA 分支
  -> reshape 为 [B*F, 3, 224, 224] 后逐帧抽特征
  -> Swin-T(IMAGENET1K_V1): patch embedding + shifted-window stages + final LayerNorm
  -> 空间池化得到帧特征 [B, F, 768]
  -> 位置编码 + TransformerEncoder 时序融合
  -> 时序平均
  -> 质量预测头
  -> 质量分数 [B]
```

### 支持的配置

| 输入 | 张量形状 | 主干网络 | 归一化 |
| :--- | :--- | :--- | :--- |
| 图像 | `[B, 3, H, W]` | ResNet50 ImageNet backbone (`BatchNorm2d`) | RGB、224x224、`[0, 1]`、ImageNet mean/std |
| 视频 | `[B, F, 3, H, W]` | Swin-T ImageNet backbone (`LayerNorm`) + Transformer 时序融合 | RGB 帧、224x224、`[0, 1]`、ImageNet mean/std |

当前模型不再保留旧 Swin 兼容 shim。检查点应由当前 `IQAVQANet` 实现生成。

### 损失函数：任务感知混合损失

`IQAVQALoss` 包含三个部分，根据 `task_type`（`iqa` 或 `vqa`）赋予不同的权重：

```text
总损失 = w_mse × MSE + w_rank × RankLoss + w_plcc × (1 − PLCC)
```

| 任务 | MSE | Rank | PLCC |
| :--- | :-- | :--- | :--- |
| IQA (`resnet_iqa`) | 0.7 | 0.3 | 0.0 |
| VQA (`timeswin_vqa`) | 0.4 | 0.3 | 0.3 |

- **MSE 损失 (MSE Loss)**：绝对预测准确度
- **排序损失 (Rank Loss)**：成对排序一致性（采用采样策略，限制最大配对数为 `max_pairs`）
- **PLCC 损失 (PLCC Loss)**：`1 − Pearson 相关系数`；在 VQA 任务中引入该项，旨在直接优化与人类 MOS（平均主观评分）的线性对齐程度

---

## 训练流程 <a id="training-pipeline"></a>

### 快速开始训练

#### 第 1 步：环境配置

```bash
# 初始化环境并安装依赖
make install

# 检查环境状态
make info

# 下载数据集、解压（unrar）并创建符号链接
make data
```

#### 第 2 步: 训练命令

直接使用 `uv` 运行训练：

```bash
# TID2013 (图像质量评估)
uv run python -m src.main --model resnet_iqa --dataset tid2013

# KoNViD-1k (视频质量评估)
uv run python -m src.main --model timeswin_vqa --dataset konvid-1k

# T2VQA-DB (文生视频质量评估)
uv run python -m src.main --model timeswin_vqa --dataset t2vqa-db
```

训练入口按以下顺序执行：

```text
完整性检查 -> EDA/统计分析 -> 分组 train/val/test 划分 -> 解码时应用 ImageNet 预处理的分组 K 折训练 -> 训练图表 -> 检查点部署
```

*注意：默认情况下，`make` 命令使用 `DEBUG=0`。如有需要，可通过追加 `DEBUG=1` 来覆盖此设置。*

> [!NOTE]
> 目前仅提供两种模型配置：`resnet_iqa`（图像/ResNet50）和 `timeswin_vqa`（视频/Swin-T）。模型配置会自动从 `config/models/*.yaml` 加载——只需在该目录下放入新的 YAML 文件（例如 `resnet_vqa.yaml`）即可注册新的配置组合，随后即可在命令中引用。

> [!NOTE]
> `scripts/setup_env.sh` 现在会安装并验证 `hatchling`，因此 `deploy/` 可以直接通过 `pyproject.toml` 完成构建，无需额外手动配置。

> [!WARNING]
> `--skip_integrity` 只建议用于快速调试。正常训练应保留完整性检查，避免缺失或损坏样本进入交叉验证。

### 进阶选项

您可以使用以下训练和调试模式来扩展框架功能：

| 模式 | 适用场景 | uv / Shell 命令 |
| :----- | :--------- | :------------------- |
| **冒烟测试 (Smoke Test)** | 快速功能检查 | `uv run python -m src.main --smoke_test` |
| **调试模式 (Debug Mode)** | 启用断点和详细日志 | `LOG_LEVEL=DEBUG uv run python -m src.main` |
| **后台运行 (Background)** | 在远程服务器上持久运行 | `nohup uv run python -m src.main > results/scripts_logs/train.log 2>&1 &` |

> [!TIP]
> 使用以下命令监控实时训练进度：
>
> ```bash
> tail -f results/scripts_logs/train.log
> ```

---

## 评估与指标 <a id="evaluation-metrics"></a>

### 核心指标

| 指标 | 全称 | 解读 |
| :--- | :--- | :--- |
| **PLCC** | Pearson 线性相关系数 | 线性关系（准确度） |
| **SROCC** | Spearman 等级相关系数 | 单调关系（排序） |
| **KROCC** | Kendall 等级相关系数 | 序数一致性 |
| **RMSE** | 均方根误差 | 预测误差幅度 |
| **R²** | 决定系数 | 可解释方差 |

### 可视化图表

该框架会自动生成：

- **EDA 分布图**：MOS 直方图与箱线图

- **训练历史**：损失曲线、PLCC/SROCC 变化趋势

- **残差分析**：散点图、误差分布

- **Fold 汇总**：每折 PLCC/SROCC/RMSE/R² 汇总与稳定性视图

- **跨模型/跨折比较**：基于已有 fold history 生成对比柱状图

输出路径：`results/{dataset}/plots/`

EDA 图表保存在 `results/{dataset}/eda/`。所有 `{dataset}` 输出目录统一使用数据集注册表中的小写 key，例如 `results/tid2013/` 和 `results/konvid-1k/`。

---

## 部署与推理 API <a id="deployment-api"></a>

部署侧现在拆分为 `deploy/api.py`（FastAPI 服务）、`deploy/cli.py`（批量推理 CLI）和 `deploy/core/`（基于 Pydantic 的运行时配置、预处理、权重加载与推理辅助函数）。两个入口共享同一套运行时默认配置。

### 目录结构

```text
deploy/
├── api.py               # FastAPI 服务（本文件）
├── cli.py               # 图像/视频批量推理 CLI
├── core/                # 运行时配置、预处理、加载与推理辅助函数
├── iqa-models/
│   └── tid2013_best.pt   # 默认 IQA 检查点 (resnet_iqa，基于 TID2013 训练)
└── vqa-models/
    └── konvid-1k_best.pt # 默认 VQA 检查点 (timeswin_vqa，基于 KoNViD-1k 训练)
```

训练成功后，当前 run 中表现最好的检查点会自动复制到部署目录：

```text
IQA -> deploy/iqa-models/{dataset}_best.pt
VQA -> deploy/vqa-models/{dataset}_best.pt
```

### 启动服务

```bash
uv run python -m deploy.api
```

该服务监听 `0.0.0.0:8000` 端口，并在启动时预先加载两个检查点；如果未找到任何模型文件，启动将失败并抛出 `RuntimeError: No models loaded`。

### 接口 (Endpoints)

| 接口路径 | 方法 | 用途 |
| :--- | :--- | :--- |
| `/health` | GET | 返回已加载的模型 ID 和推理设备 |
| `/evaluate` | POST | 对上传的图像/视频进行推理，并返回统一的 MOS 分数 |

`/evaluate` 接口接受 `multipart/form-data` 格式的请求，包含以下字段：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `file` | file | 待评分的图像或视频 |
| `media_type` | string | `"image"` 或 `"video"` |
| `task_type` | string | 原样返回在响应中，不用于路由选择 |
| `model` | string | `"resnet_iqa"` 或 `"timeswin_vqa"` —— 指定由哪个已缓存模型处理请求 |

支持双向跨模态推理：当输入为视频时，`resnet_iqa` 会对采样帧的预测结果取平均值；当输入为图像时，`timeswin_vqa` 会将单张图像扩展为伪视频（`image_to_video_tensor`）。

响应包含 `mos`（统一的 0–5 分制，计算方式为 `raw_score × 5`）和 `dataset_mos`（反归一化回源数据集原始 MOS 范围的分数，用于调试）。

> [!NOTE]
> 对于使用 v0.5.0 及以上版本保存的检查点（checkpoint），MOS 范围（`mos_min`/`mos_max`）存储在配置中，并在加载时恢复。反归一化过程是自动进行的。

> [!TIP]
> `dataset_mos` 字段返回的是基于源数据集原始 MOS 范围的分数（例如 TID2013 为 0-9，KoNViD-1k 为 1-5），而 `mos` 字段则统一归一化为 0-5 分制，以便进行跨数据集比较。

### 批量 CLI

```bash
uv run python -m deploy.cli -i examples/images/
uv run python -m deploy.cli -i examples/videos/
uv run python -m deploy.cli -i examples/
```

CLI 会自动识别图像和视频文件，并将结果写入 `reports/iqa-test/` 或 `reports/vqa-test/`；`make test-images`、`make test-videos` 和 `make test-all` 目标就是调用它。

---

## 项目主要结构 <a id="project-main-structure"></a>

```text
deep-vqa-framework/
├── Makefile                # 自动化与工作流命令
├── README.md               # 项目概览
├── DISCLAIMER.md           # 法律责任与资源使用政策
├── pyproject.toml          # 依赖、环境与构建管理 (uv + hatchling)
│
├── config/                 # YAML 配置文件（用户可编辑）
│   ├── basic.yaml            # 系统与训练的全局默认设置
│   ├── dataset_config.yaml   # 数据集特定元数据
│   └── models/                 # 模型架构参数
│
├── datasets/                 # 数据存储与符号链接路由
│   ├── KoNViD-1k/               # 视频质量数据集
│   ├── T2VQA-DB/                 # 文本到视频问答 (T2VQA) 数据集
│   └── TID2013/                  # 图像质量数据集
│
├── docs/                     # 交互式架构图与使用手册
│   ├── pipeline.html            # 系统执行流程与模块流转
│   └── Cloud_Platform_Rental_Guide.md
|
├── reports/                  # pip-audit、SBOM、safety 的安全报告
|
├── results/
|   ├── {dataset}/
│   │   ├── train_logs/           # 训练历史记录、CSV 日志
│   │   ├── plots/                # 损失曲线、残差图
│   │   ├── eda/                  # 数据集分析图表
│   │   ├── model_outputs/        # .pt 文件
│   │   └── corrupted/            # 隔离的损坏媒体文件与被拒绝标签备份
│   └── scripts_logs/             # Shell 脚本日志 (setup, data, etc.)
|
├── docker/                   # 容器配置
│   ├── docker-compose.yaml      # 主 compose 配置
│   ├── docker-compose.docker.yaml # Docker GPU 支持
│   └── docker-compose.podman.yaml # Podman GPU 支持
│
├── .github/workflows/        # CI/CD 流水线
│   └── ci.yaml                 # 持续集成
|
├── scripts/                  # 基础设施自动化脚本
│   ├── bootstrap.sh             # 系统级初始化（apt、镜像源、系统工具）
│   ├── setup_env.sh             # 项目级初始化（uv、.venv、Python 依赖、hatchling 安装/验证）
│   ├── manage_data.sh           # 数据下载与预处理
│   ├── archive_results.sh       # 结果打包归档
│   ├── cache_clean.sh           # 缓存清理
│   └── ci_test_extract.sh       # smart_extract 测试的 CI 辅助脚本
│
├── deploy/                  # 独立推理服务与批量 CLI（与训练解耦）
│   ├── api.py                    # FastAPI 服务
│   ├── cli.py                    # 图像/视频批量推理 CLI
│   ├── core/                     # 运行时配置、预处理、加载与推理辅助函数
│   ├── iqa-models/               # 由 api.py 提供的 IQA .pt 模型权重
│   └── vqa-models/               # 由 api.py 提供的 VQA .pt 模型权重
│
└── src/                       # 核心框架逻辑
    ├── main.py                   # 全局执行入口
    ├── core/                        # 训练引擎与评估流程
    ├── data/                        # 数据加载器、预处理、EDA（探索性数据分析）与完整性分析
    ├── models/                      # Backbones、heads、losses、metrics 与 IQAVQANet
    ├── utils/                        # 配置、日志记录与路径管理
    └── config/                     # Pydantic 配置系统（代码实现）
```

---

## Docker / Podman 支持 <a id="docker-support"></a>

该框架支持使用 Docker 和 Podman 进行容器化开发与部署。

### Docker 开发部署方法

```bash
# 构建并进入开发容器
make docker-dev

# 在容器内运行训练
make docker-train

# 启动推理 API 服务
make docker-infer

# 停止所有容器
make docker-stop

# 对示例图像做批量推理
make test-images

# 对示例视频做批量推理
make test-videos

# 对全部示例做批量推理
make test-all

# 清理容器、镜像、卷和网络
make docker-purge-all

# 检查容器环境
make docker-manage
```

### 容器配置

| 组件 | 描述 |
| :--- | :--- |
| `Dockerfile` | 多阶段构建：`base`（共享依赖）、`train`（训练）、`prod`（推理） |
| `docker-compose.yaml` | 主 Compose 配置文件，启用 `json-file` 日志轮转并挂载 `.cache` |
| `docker-compose.docker.yaml` | Docker 专用 GPU 支持，并为构建阶段启用 host 网络 |
| `docker-compose.podman.yaml` | Podman 专用 GPU 支持，并为构建阶段启用 host 网络 |

Makefile 会自动识别 Docker 或 Podman。Podman 用户直接运行 `make docker-*` 即可，不需要设置 `alias docker=podman`；只有手动运行容器命令时才可能需要 alias。

数据集脚本会检测 `http_proxy`/`HTTP_PROXY`。在 AutoDL 云 GPU 实例上，下载数据集前可以先启用平台代理：

```bash
source /etc/network_turbo
```

容器内的 torch/uv 缓存挂载到 `/app/.cache`。运行服务会设置 `XDG_CACHE_HOME=/app/.cache`、`TORCH_HOME=/app/.cache/torch` 和 `UV_CACHE_DIR=/app/.cache/uv`，因此已下载的 torchvision backbone 可以复用。

如果宿主机代理绑定在 `127.0.0.1`，需要区分运行阶段和构建阶段：服务的 `network_mode: host` 只作用于运行容器，Dockerfile 的 `RUN` 步骤需要 build network。compose overlay 已设置 `build.network: host`；若要将代理变量传入 Dockerfile 构建，运行：

```bash
make docker-train BUILD_ARGS='--build-arg USE_BUILD_PROXY=true'
```

---

## 系统概览 <a id="system-overview"></a>

为了帮助您快速理解系统架构与执行流程，我们提供了**交互式流水线可视化图表**。

[**→ 打开交互式架构图**](docs/pipeline.html)

该图表展示了：
- 数据在系统各阶段的流转方式
- 关键组件及其交互关系
- 整个工作流的执行顺序

> 图表使用 Mermaid 绘制，支持在浏览器中交互式查看

---

## 配置指南 <a id="configuration-guide"></a>

配置由 `load_config()` 函数组装，并返回一个 Pydantic `Config` 对象。
所有设置均在加载时进行验证和类型检查。
路径通过 `cfg.paths.xxx_dir(dataset_name)` 方法进行解析。

| 阶段 | 文件 | 合并方式 |
| ------- | ------ | --------- |
| 1 (基础) | `basic.yaml` | 首先加载作为初始配置 |
| 2 (模型) | `models/{model}.yaml` | 在阶段 1 基础上进行深度合并（匹配的键会被覆盖） |
| 3 (数据集) | `dataset_config.yaml` | **未合并至顶层键** — 匹配的数据集条目将作为 `config["dataset"]` 整体附加 |

合并后的结果会在构造 `Config(**merged)` 时由 Pydantic 校验。路径解析由 Pydantic 的 `PathsConfig` 及其类型化方法处理。

---

## 故障排查 <a id="troubleshooting"></a>

### CUDA 显存溢出 (OOM)

当发生 OOM 时，请调整模型 YAML 配置文件（例如 `config/models/timeswin_vqa.yaml`）中的以下参数：

**降低显存占用：**

- 减小 `preprocessing.batch_size` — 降低单批次显存占用
- 减小 `model.num_frames` — 减少待处理的视频帧数
- 切换到更轻量级的 `model.backbone`（例如使用 `resnet50` 替代 `swin_t`）
- 减少 `model.transformer_layers` — 降低时间维度融合的深度

**补偿较小的批次大小：**

- 增加 `train.gradient_accumulation_steps` — 保持有效批次大小（即 `batch_size × gradient_accumulation_steps`）
- 确保设置 `amp: true` — 混合精度训练可显著降低显存占用

> [!NOTE]
> `IQAVQANet` 尚未实现梯度检查点（gradient checkpointing）功能 — 请勿在配置中设置 `gradient_checkpointing: true`，该设置目前无效。

### 视频加载后端（AutoDL 特有）

> [!WARNING]
在 AutoDL 或类似的云端 GPU 实例上，OpenCV 的 `VideoCapture` 可能会因缺少系统依赖项而失败。

解决方案：使用 Decord

```bash
uv add decord
```

Decord 已预配置为默认后端。如果 Decord 不可用，框架会自动回退到 OpenCV，但在 AutoDL 上这种回退可能会失败。在 AutoDL 上进行视频训练时，请务必使用 Decord。

### 训练速度慢

| 问题 | 优化方案 |
| :--- | :--- |
| 数据加载瓶颈 | 增加 `num_workers`（例如设为 8） |
| Batch size（批次大小）过小 | 使用 `gradient_accumulation_steps` |
| 视频解码缓慢 | 确保已安装 Decord |

### 磁盘空间耗尽

`results/{dataset}/corrupted/` 目录存储了由 `DataEDA.check_integrity()` 移出的文件。
- 若不再需要，可手动清理
- 或在 `scripts/cache_clean.sh` 中添加清理步骤

> [!TIP]
> `results/{dataset}/corrupted/labels/` 会保存缺失或损坏样本对应标签的 CSV/JSONL 备份。重复的 `sample_id` 会作为 `repeated_sample_ids` 报告并保留，因为 IQA/VQA 数据集中同一内容可能合法对应多个失真样本或多个标签。

### 训练无报错挂起（后台运行）

如果使用 `nohup` 或在后台运行的训练任务看似卡住且没有新的日志输出，请检查 `src/main.py` 中是否存在 `pdb.set_trace()` 断点。这些断点会在异常发生时触发并等待标准输入（stdin），从而阻塞非交互式进程。

**解决方案：**
- 调试时以交互方式（不使用 `nohup`）运行训练
- 检查 `results/{dataset}/train_logs/*.log` 以获取错误详情
- 终止进程并修复根本问题后再重试

---

## 依赖项安全 <a id="dependency-security"></a>

该框架包含用于审计依赖项的安全工具：

| 命令 | 用途 |
| :--- | :--- |
| `make vuln-audit` | 扫描依赖项中的已知漏洞 |
| `make sbom` | 生成软件物料清单 (SBOM/CycloneDX) |
| `make safety` | 使用 Safety 检查依赖项（旧版工具，需登录） |
| `make security-all` | 运行所有安全检查 |

> [!NOTE]
> `pip-audit` 是主要的漏洞扫描工具。`safety` 工具需要注册或登录。

## 📄 许可证 <a id="license"></a>

- **框架**: [MIT](LICENSE)
- **作者**: [@autentisitet](https://github.com/autentisitet)
- **版本**: 0.6.4

---

## 👥 贡献者 <a id="contributors"></a>

| 姓名 | 角色 | 贡献内容 |
| :--- | :--- | :--- |
| **[@autentisitet](https://github.com/autentisitet)** | 项目负责人 / 核心开发者 | 框架架构、训练流水线、推理引擎、Docker/Podman 容器化、多阶段构建、推理 API 设计 |
| **[@yss0120](https://github.com/yss0120)** | 前端开发者 | 交互式 UI/UX (`index.html`)、主观盲测评分系统、质量概览（Quality Passport）可视化 |
| **[@Zed-23](https://github.com/Zed-23)** | DevOps 与质量保证 | CI/CD 流水线、自动化测试与冒烟测试、Shell 脚本修复、CUDA 显存溢出 (OOM) 调试 |
| **[@bazhina-5566](https://github.com/bazhina-5566)** | 后端 API 开发者 | FastAPI 服务 (`deploy/api.py`)、模型检查点加载、部署 API 设计 |

> [!NOTE]
> 欢迎贡献代码！请参阅 [CONTRIBUTING.md](.github/CONTRIBUTING.md) 了解贡献指南。

---

## 🙏 致谢 <a id="acknowledgments"></a>

- PyTorch 团队（提供深度学习框架）
- Decord 开发者（提供高效视频加载功能）
- FastAPI（提供生产级 API 框架）
- TID2013、KoNViD-1k、T2VQA-DB 数据集提供方

---

## ⚖️ 法律声明与免责条款
有关第三方工具使用、数据集合规性及资源使用的详细信息，请参阅 [DISCLAIMER.md](DISCLAIMER.md) 文件。

---

如需了解详细的贡献指南和问题反馈流程，请查看 `.github` 文件夹。

**专为研究社区打造 ❤️**
