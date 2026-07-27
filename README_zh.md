# deep-vqa-framework

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub release](https://img.shields.io/github/v/release/autentisitet/deep-vqa-framework?include_prereleases)](https://github.com/autentisitet/deep-vqa-framework/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.4.5--beta-blue.svg)](https://github.com/autentisitet/deep-vqa-framework)
[![Code Quality: ruff+black+isort+mypy](https://img.shields.io/badge/code%20quality-ruff%2Bblack%2Bisort%2Bmypy-4B8BBE.svg)](https://github.com/autentisitet/deep-vqa-framework)
[![Security: pip-audit+sbom](https://img.shields.io/badge/security-pip--audit%2Bsbom-9cf.svg)](https://github.com/autentisitet/deep-vqa-framework)

**🌐 [English](README.md) | [简体中文](README_zh.md)**

**一个用于图像质量评估 (IQA) 和视频质量评估 (VQA) 的统一深度学习框架。**

该框架为质量评估模型的训练、评估和部署提供了一站式（端到端）解决方案。它采用统一架构，能够无缝处理图像和视频输入，并支持多数据集、交叉验证流程以及生产级推理 API。

> [!NOTE]
> 本框架主要在 AutoDL 云 GPU 实例上进行了测试。
> 您可以运行以下命令，在 AutoDL 云实例上开启代理：

```bash
source /etc/network_turbo
```

---

## 目录

- [架构与设计决策](#architecture-decisions)
- [模型架构](#model-architecture)
- [训练流水线](#training-pipeline)
- [评估与指标](#evaluation-metrics)
- [部署与推理 API](#deployment-api)
- [项目主要结构](#project-main-structure)
- [系统概览](#system-overview)
- [配置指南](#configuration-guide)
- [故障排查](#troubleshooting)
- [依赖项安全](#dependency-security)
- [许可证](#license)
- [贡献者](#contributors)
- [致谢](#acknowledgments)

---

## 架构与设计决策 <a id="architecture-decisions"></a>

### 统一的 IQA/VQA 架构

该框架实现了一个维度感知（dimension-aware）的路由系统，可在图像（4D 张量）和视频（5D 张量）处理模式之间自动切换。

**关键设计决策：**

| 决策 | 实现方式 | 理由 |
| :--- | :--- | :--- |
| **统一模型** | 单个 `IQAVQANet` 同时处理 4D 和 5D 输入 | 消除重复代码，确保质量指标的一致性 |
| **灵活的主干网络** | Swin-T / ResNet50 配合自动特征适配 | 平衡准确率与内存消耗 |
| **时序融合** | 使用 Transformer 编码器进行视频帧聚合 | 捕捉帧间的长距离依赖关系 |
| **任务感知损失函数** | MSE + Rank + PLCC，根据 `task_type` 重新加权 | 优化绝对预测、相对排序及线性 MOS 对齐 |
| **多数据集支持** | 基于 YAML 的配置与工厂模式 | 无需修改代码即可轻松添加新数据集 |
| **路径抽象** | 基于 DSL 的 `PathManager` 与 YAML 路由 | 消除硬编码路径，支持符号链接 |
| **延迟资源解析** | 带有预构建索引的 `CaseInsensitiveAssetResolver` | O(1) 文件查找，支持不区分大小写的匹配 |

---

## 模型架构 <a id="model-architecture"></a>

### IQAVQANet：统一质量评估网络

```python
# 架构概览
输入 (4D: [B,3,H,W] 或 5D: [B,F,3,H,W])
↓
骨干网络 (Swin-T / ResNet50)
↓
空间池化 (AdaptiveAvgPool2d)
↓
[时序融合] ← TransformerEncoder (仅限视频)
↓
质量评估头 (3层 MLP + Sigmoid)
↓
输出：质量分数 (范围 0-1)
```

### 支持的配置

| 骨干网络 | 参数量 | IQA | VQA | 显存占用 (单样本) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | 25M | ✅ | ✅ | ~2GB (8 帧) |
| **Swin-T** | 28M | ✅ | ✅ | ~4GB (8 帧) |

### 损失函数：任务感知混合损失

`IQAVQALoss` 包含三个部分，根据 `task_type`（`iqa` 或 `vqa`）赋予不同的权重：

```text
总损失 = w_mse × MSE + w_rank × RankLoss + w_plcc × (1 − PLCC)
```

| 任务 | MSE | Rank | PLCC |
| :--- | :-- | :--- | :--- |
| IQA (`resnet_iqa`) | 0.7 | 0.3 | 0.0 |
| VQA (`timeswin_vqa`) | 0.4 | 0.3 | 0.3 |

- **MSE Loss**：绝对预测准确度
- **Rank Loss**：成对排序一致性（采用采样方式，限制最大配对数为 `max_pairs`）
- **PLCC Loss**：`1 − Pearson 相关系数`；在 VQA 任务中引入该项权重，旨在直接优化与人类 MOS（平均主观评分）的线性对齐程度

> [!WARNING]
> 已知问题：目前 `mode` 参数未从训练引擎传递至 `IQAVQALoss.forward()`，因此在完成参数传递对接之前，VQA 任务将默认使用 IQA 的权重配置。

---

## 训练流程 <a id="training-pipeline"></a>

### 快速开始训练

#### 第 1 步：环境设置

```bash
# 初始化环境并安装依赖
make setup

# 检查环境状态
make info

# 下载数据集、解压并创建符号链接
make data
```

#### 第 2 步：训练命令

直接使用 `uv` 运行训练：

```bash
# TID2013 (图像质量评估)
uv run python -m src.main --model resnet_iqa --dataset tid2013

# KoNViD-1k (视频质量评估)
uv run python -m src.main --model timeswin_vqa --dataset konvid-1k

# T2VQA-DB (文生视频质量评估)
uv run python -m src.main --model timeswin_vqa --dataset t2vqa-db
```

*注意：默认情况下，`make` 命令使用 `DEBUG=0`。如有需要，可通过追加 `DEBUG=1` 来覆盖此设置。*

> [!NOTE]
> 目前仅提供两种模型配置：`resnet_iqa` (图像/ResNet50) 和 `timeswin_vqa` (视频/Swin-T)。模型配置会自动从 `config/models/*.yaml` 加载——只需在该目录下放入新的 YAML 文件（例如 `resnet_vqa.yaml`）即可注册新的组合，随后即可在命令中引用。

### 进阶选项

您可以使用以下训练和调试模式来扩展框架功能：

| 模式 | 适用场景 | uv / Shell 命令 |
| :----- | :--------- | :------------------- |
| **冒烟测试** | 快速功能检查 | `uv run python -m src.main --smoke_test` |
| **调试模式** | 启用断点和详细日志 | `LOG_LEVEL=DEBUG uv run python -m src.main` |
| **后台运行** | 在远程服务器上持续运行 | `nohup uv run python -m src.main > results/scripts_logs/train.log 2>&1 &` | > [!TIP]
> 使用以下命令监控实时训练进度：
>
> ```bash
> tail -f results/scripts_logs/train.log
> ```

---

## 评估与指标 <a id="evaluation-metrics"></a>

### 核心指标

| 指标 | 全称 | 含义 |
| :--- | :--- | :--- |
| **PLCC** | Pearson Linear Correlation Coefficient | 线性关系（准确度） |
| **SROCC** | Spearman Rank Order Correlation Coefficient | 单调关系（排序） |
| **KROCC** | Kendall Rank Correlation Coefficient | 序数一致性 |
| **RMSE** | Root Mean Square Error | 预测误差幅度 |
| **R²** | Coefficient of Determination | 可解释方差 |

### 可视化

该框架会自动生成：

- **训练历史**：损失曲线、PLCC/SROCC 变化趋势

- **残差分析**：散点图、误差分布

- **跨模型比较**：多模型对比柱状图

输出路径：`results/{dataset}/plots/`

---

## 部署与推理 API <a id="deployment-api"></a>

一个独立的 FastAPI 服务（`deploy/api.py`）提供已训练模型检查点（checkpoint）的推理接口，与训练流程解耦。该服务在启动时加载一个 IQA 模型和一个 VQA 模型，并统一输出 0-5 范围内的 MOS（平均主观意见分）评分，无论具体由哪个模型处理请求。

### 目录结构

```text
deploy/
├── api.py               # FastAPI 服务（本文件）
├── infer.py              # 预处理 + 检查点加载 + 推理辅助函数
├── iqa-models/
│   └── tid2013_best.pt   # 默认 IQA 检查点 (resnet_iqa，基于 TID2013 训练)
└── vqa-models/
└── konvid_best.pt    # 默认 VQA 检查点 (timeswin_vqa，基于 KoNViD-1k 训练)
```

> [!WARNING]
> 检查点路径是相对于 `api.py` 所在位置解析的（即 `Path(__file__).resolve().parent / "iqa-models" / ...`），因此 `.pt` 文件必须放置在 `deploy/iqa-models/` 和 `deploy/vqa-models/` 目录下，而不能放在 `results/model_outputs/` 中。

### 启动服务

```bash
cd deploy
uv run python -m deploy.api
```

该服务监听 `0.0.0.0:8000` 端口，并在启动时预先加载两个检查点；如果未找到任何模型文件，启动将失败并抛出 `RuntimeError: 没有成功加载任何模型，服务启动失败`。

### API 接口

| 接口路径 | 方法 | 用途 |
| :--- | :--- | :--- |
| `/health` | GET | 返回已加载的模型 ID 和推理设备信息 |
| `/evaluate` | POST | 对上传的图像/视频进行推理，并返回统一的 MOS 评分 |

`/evaluate` 接受 `multipart/form-data` 格式，包含以下字段：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `file` | 文件 | 待评分的图像或视频 |
| `media_type` | 字符串 | `"image"` 或 `"video"` |
| `task_type` | 字符串 | 原样返回在响应中，不用于路由 |
| `model` | 字符串 | `"resnet_iqa"` 或 `"timeswin_vqa"` — 选择处理请求的缓存模型 |

支持双向跨模态推理：当输入为视频时，`resnet_iqa` 会对采样帧的预测结果取平均值；当输入为图像时，`timeswin_vqa` 会将单张图像扩展为伪视频（`image_to_video_tensor`）。

响应包含 `mos`（统一的 0–5 分制，计算方式为 `raw_score × 5`）和 `dataset_mos`（反归一化回源数据集原始 MOS 范围的值，用于调试）。

> [!NOTE]
> MOS 反归一化优先使用检查点（checkpoint）保存的配置中的 `dataset_info.mos_min`/`mos_max`；若缺失，则回退使用硬编码常量（`DATASET_MOS_PARAMS`，仅适用于 TID2013/KoNViD-1k）。鉴于目前的训练流程无法保证 `dataset_info` 中包含 `mos_min`/`mos_max`（参见上文关于 `dataset_info` 的配置指南说明），实际运行中很可能使用的是硬编码的回退值——因此，在信任 `dataset_mos` 输出之前，请务必核实启动日志中打印的范围（`✅ ... 模型加载完成 (... MOS 范围: X~Y)`）是否与您的数据集相符。

> [!NOTE]
> 目前 CORS 设置为完全开放（`allow_origins=["*"]`）——这适用于本地开发，但在将服务暴露给本机以外的环境之前，请务必收紧此设置。 ---

## 项目主要结构 <a id="project-main-structure"></a>

```text
deep-vqa-framework/
├── Makefile                # 自动化与工作流命令
├── README.md               # 项目概览
├── DISCLAIMER.md           # 法律责任与资源使用政策
├── pyproject.toml          # 依赖与环境管理 (uv)
│
├── config/                 # YAML 配置文件模块
│   ├── paths.yaml            # 路径解析 DSL（根目录与解析器）
│   ├── basic.yaml            # 系统与训练的全局默认设置
│   ├── dataset_config.yaml   # 数据集特定元数据
│   └── models/                 # 模型架构参数
│
├── datasets/                 # 数据存储与符号链接路由
│   ├── KoNViD-1k/               # 视频质量数据集
│   ├── T2VQA-DB/                 # 文本到视频问答 (T2VQA) 数据集
│   └── TID2013/                  # 图像质量数据集
│
├── docs/                     # 交互式架构图与手册
│   ├── pipeline.html            # 系统执行与模块流程
│   └── Cloud_Platform_Rental_Guide.md
│
├── results/                  # 全局输出与日志
│   ├── model_outputs/           # 训练检查点 (checkpoints)
│   ├── scripts_logs/
│   └── plots/                     # 可视化图表（损失、残差等）
│
├── scripts/                  # 基础设施自动化脚本
│   ├── manage_data.sh           # 下载与数据准备
│   ├── setup_env.sh              # 环境与系统初始化
│   ├── archive_results.sh         # 结果打包归档
│   └── *.sh                        # 辅助维护与清理脚本
│
├── deploy/                  # 独立推理服务（与训练解耦）
│   ├── api.py                    # FastAPI 服务 —— 模块命名见下方说明
│   ├── infer.py                   # 预处理 + 加载检查点 + 预测
│   ├── iqa-models/                # 由 api.py 提供的 IQA .pt 检查点
│   └── vqa-models/                # 由 api.py 提供的 VQA .pt 检查点
│
└── src/                       # 核心框架逻辑
    ├── main.py                   # 全局执行入口点
    ├── core/                        # 训练引擎与评估流程
    ├── data/                       # 数据加载器、EDA（探索性数据分析）及完整性分析
    ├── models/                      # 架构定义 (IQAVQA-Net)
    └── utils/                        # 配置、日志记录及路径管理
```

---

## 系统概览 <a id="system-overview"></a>

为了帮助您快速掌握系统架构和执行流程，我们提供了**交互式流水线可视化图**。

[**→ 打开交互式架构图**](docs/pipeline.html)

该图展示了：
- 数据如何在系统各阶段间流动
- 关键组件及其交互方式
- 整个工作流的执行顺序

---

## 配置指南 <a id="configuration-guide"></a>

### 配置分层机制

配置由 `config_loader.load_system_config()` 分两个步骤组装而成：

| 阶段 | 文件 | 合并方式 |
| ------- | ------ | --------- |
| 1 (基础) | `basic.yaml` | 首先加载，作为初始配置 |
| 2 (模型) | `models/{model}.yaml` | 在阶段 1 基础上进行深度合并（匹配的键会覆盖原有值） |
| 3 (数据集) | `dataset_config.yaml` | **不合并到顶层键中** —— 匹配的数据集条目将作为 `config["dataset_info"]` 整体附加 |

合并后的结果在训练前会通过 `config_loader.validate_config_schema()` 进行必填字段验证。路径解析（`config/paths.yaml`）由 `PathManager` 单独处理，不包含在此次合并中。

### 视频训练的内存优化

```yaml
# 如果遇到 CUDA 显存不足 (OOM)
preprocessing:
batch_size: 1              # 减小批次大小 (batch size)
num_workers: 0             # 禁用多进程

model:
num_frames: 4              # 减少时间维度帧数
backbone: "resnet50"       # 使用更小的骨干网络 (backbone)
transformer_layers: 1      # 降低 Transformer 深度

train:
gradient_accumulation_steps: 4  # 模拟更大的批次
amp: true                  # 启用混合精度训练 (AMP)
```

---

## 故障排查 <a id="troubleshooting"></a>

### CUDA 显存不足 (OOM)

| 现象 | 解决方案 |
| :--- | :--- |
| 第一个批次即 OOM | 将 `batch_size` 减小为 1 |
| 运行几个 epoch 后 OOM | 减少 `num_frames` 或切换到 `resnet50` 骨干网络 |
| 验证阶段 OOM | 将 `num_frames` 减小为 4 |

> [!NOTE]
> `IQAVQANet` 尚未实现梯度检查点（gradient checkpointing）功能 —— 请勿在配置中设置 `gradient_checkpointing: true`，该设置目前无效。

### 视频加载后端（AutoDL 专用）

> [!WARNING]
在 AutoDL 或类似的云端 GPU 实例上，OpenCV 的 `VideoCapture` 可能会因缺少系统依赖项而失败。

解决方案：使用 Decord

```bash
uv add decord
```

Decord 已预配置为默认后端。如果不可用，框架会自动回退到 OpenCV，但在 AutoDL 上这种回退可能会失败。在 AutoDL 上进行视频训练时，请务必使用 Decord。

### 训练速度慢

| 问题 | 优化方案 |
| :--- | :--- |
| 数据加载瓶颈 | 增加 `num_workers: 8` |
| 小批量（batch size） | 使用 `gradient_accumulation_steps` |
| 视频解码缓慢 | 确保已安装 Decord |

### 磁盘空间耗尽

`quarantine/` 目录（由 `DataEDA.check_integrity()` 在移动损坏文件时创建）目前未被 `cache_clean.sh` 脚本处理 —— 请手动清理，或在脚本中添加清理步骤。

### 训练无报错挂起（后台运行）

如果使用 `nohup` 或在后台运行的训练任务看似卡住（无新日志输出且未崩溃），请检查是否停在了 `pdb.set_trace()` 断点处。`PathManager.resolve(..., mkdir=True)` 在创建目录时遇到 `PermissionError` 或 `OSError` 会进入 `pdb` 调试模式；对于非交互式进程，这会导致程序静默阻塞（等待标准输入），而不是抛出异常。若发生此情况，请终止进程并检查磁盘权限或空间。

---

## 依赖项安全 <a id="dependency-security"></a>

该框架包含用于审计依赖项的安全工具：

| 命令 | 用途 |
| :--- | :--- |
| `make vuln-audit` | 扫描依赖项以查找已知漏洞 |
| `make sbom` | 生成软件物料清单 (SBOM) (CycloneDX 格式) |
| `make safety` | 使用 Safety 检查依赖项（旧版工具，需登录） |
| `make security-all` | 运行所有安全检查 |

> [!NOTE]
> `pip-audit` 是主要的漏洞扫描工具。`safety` 工具需要注册或登录。

---

## 📄 许可证 <a id="license"></a>

- **框架**: [MIT](LICENSE)
- **作者**: [@autentisitet](https://github.com/autentisitet)
- **版本**: 0.4.5-beta (预发布版)

---

## 👥 贡献者 <a id="contributors"></a>

| 姓名 | 角色 | 贡献内容 |
| :--- | :--- | :--- |
| **[@autentisitet](https://github.com/autentisitet)** | 项目负责人 / 核心开发者 | 框架架构、训练流水线、推理引擎、部署 API |
| **[@yss0120](https://github.com/yss0120)** | 前端开发者 | 交互式 UI/UX (`index.html`)、主观盲测评分系统、质量评估概览（Quality Passport）可视化、数据探索性分析 (EDA) 流水线 |
| **[@Zed-23](https://github.com/Zed-23)** | DevOps 与质量保证 | CI/CD 流水线、自动化测试与冒烟测试、Shell 脚本修复、CUDA 显存溢出 (OOM) 调试 |
| **[@bazhina-5566](https://github.com/bazhina-5566)** | 后端 API 开发者 | FastAPI 服务 (`deploy/api.py`)、模型检查点 (checkpoint) 加载、推理 API 设计 |

> [!NOTE]
> 欢迎贡献代码！请参阅 [CONTRIBUTING.md](.github/CONTRIBUTING.md) 了解相关指南。

---

## 🙏 致谢 <a id="acknowledgments"></a>

- PyTorch 团队（提供深度学习框架）
- Decord 开发者（提供高效视频加载功能）
- FastAPI（提供生产级 API 框架）
- TID2013、KoNViD-1k 及 T2VQA-DB 数据集提供方

---

## ⚖️ 法律声明与免责声明
有关第三方工具使用、数据集合规性及资源使用的详细信息，请参阅 [DISCLAIMER.md](DISCLAIMER.md) 文件。

---

如需了解详细的贡献指南和问题反馈流程，请查看 `.github` 文件夹。

**怀着 ❤️ 为研究社区打造**
