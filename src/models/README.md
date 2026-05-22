在图像与视频质量评估（IQA/VQA）任务中，数据在网络中流转时的 Tensor 形状（Shape）变换与向量空间映射是理解整个模型的关键。

为了完美适配你的 YAML 配置（包含 `sample_frames: 16` 的视频流以及 Swin-T/ResNet50 图像流），下面我们将数据分为 **1. 纯图像模式 (IQA)** 和 **2. 视频流模式 (VQA)** 两种场景，详细拆解 Tensor 在**特征提取 $\rightarrow$ 时空融合 $\rightarrow$ 回归层**全流程中的形状退化、维数置换以及背后的数学映射逻辑。

---

### 场景一：纯图像质量评价 (IQA)

*以 `iqa_task.yaml` 中默认激活的 `backbone: "swin_t"` 为例。输入单张图像尺寸为 `[224, 224]`。*

#### 1. 特征提取阶段 (Feature Extraction)

* **Tensor 形状变换**：

$$\text{输入图像: } [B, 3, 224, 224] \longrightarrow \text{Swin-T Backbone} \longrightarrow \text{原始特征: } [B, 7, 7, 768]$$



*(如果切换为 ResNet50，原始特征形状则为 `[B, 2048, 7, 7]`)*
* **向量空间映射逻辑**：
将连续的**图像像素空间**（由高、宽、RGB通道构成的低阶物理空间）映射到**高阶语义特征空间**。
* **Swin-T** 内部将图像划分为多个 Patch，在多级 Transformer 窗口内计算自注意力。它将局部的空间像素结构（包含失真信息，如模糊、块效应）编码成一个 $768$ 维的稠密特征向量。
* 此时，空间分辨率由 $224 \times 224$ 被压缩到 $7 \times 7$，意味着空间几何信息被高度凝练，空间中的每一个点（Token）都包含了周围一个区域的质量表现。



#### 2. 时空融合阶段 (Spatial Pooling)

*由于是单张图像，此阶段退化为纯粹的“空间融合”。*

* **Tensor 形状变换**：

$$\text{原始特征: } [B, 7, 7, 768] \xrightarrow{\text{Permute(0, 3, 1, 2)}} [B, 768, 7, 7] \xrightarrow{\text{AdaptiveAvgPool2d(1)}} [B, 768, 1, 1] \xrightarrow{\text{Flatten(1)}} \text{融合向量: } [B, 768]$$


* **向量空间映射逻辑**：
将**空间分布式特征空间**（包含每个局部区域质量的特征阵列）映射到**全局图像表征空间**。
* 通过全局自适应平均池化（`AdaptiveAvgPool2d`），网络在空间维度（$7 \times 7$）上做均值计算。从物理意义上讲，这是将图像中各个局部区域的失真程度进行**全局均字化汇聚**。
* 这一步消除了空间分辨率，将图像“打包”成一个代表整张图像全局统计特征的 $768$ 维定长向量。



#### 3. 回归层阶段 (Regression Head)

* **Tensor 形状变换**：

$$[B, 768] \xrightarrow{\text{Linear(768, 512) + GELU}} [B, 512] \xrightarrow{\text{Linear(512, 256) + GELU}} [B, 256] \xrightarrow{\text{Linear(256, 1)}} \text{输出标量: } [B, 1]$$


* **向量空间映射逻辑**：
将高维的**全局图像表征空间**多级降维映射到**一维连续主观质量空间（MOS 分数空间）**。
* 通过 MLP 的非线性映射，网络将抽象的 $768$ 维特征逐步剥离、压缩，寻找特征与人类主观感知质量之间的相关性。
* 最终映射到的 $\mathbb{R}^1$ 标量空间即为你需要预测的质量打分（如 0~5 或 0~100 的连续标量值）。



---

### 场景二：真实场景视频质量评价 (VQA)

*结合 `dataset_config.yaml` 里的视频配置，假设 Batch Size 为 $B$，每个视频采样帧数 `sample_frames: 16`，主干网络仍为 `swin_t`。*

#### 1. 特征提取阶段 (Feature Extraction)

* **Tensor 形状变换**：
为了让 2D 卷积/Transformer 骨干网络能够处理视频，通常会将 Batch 维度和帧数维度合并（打包成伪图像批次）：

$$\text{视频输入: } [B, 16, 3, 224, 224] \xrightarrow{\text{View / Reshape}} \text{伪图像批次: } [B \times 16, 3, 224, 224]$$


$$\text{伪图像批次: } [B \times 16, 3, 224, 224] \longrightarrow \text{Swin-T Backbone} \longrightarrow \text{帧特征: } [B \times 16, 7, 7, 768]$$


* **向量空间映射逻辑**：
将视频在时空上的**原生帧像素空间**映射到**独立空间特征空间**。
* 此阶段骨干网络**只负责提取每帧图像的空间特征**（如第 5 帧的运动模糊、第 10 帧的噪点），帧与帧之间在此阶段还没有任何时间序列上的信息交换。



#### 2. 时空融合阶段 (Spatiotemporal Fusion)

*这是 VQA 任务的灵魂，需要将空间特征和时间线特征融为一体。*

* **第一步：空间汇聚（Spatial Pooling）**
将每帧的空间维度消除，拿到每帧的特征向量：

$$[B \times 16, 7, 7, 768] \xrightarrow{\text{Permute \& Pool \& Flatten}} [B \times 16, 768]$$


* **第二步：时序解包与维度重排（Temporal Reshape）**
将合并的维度拆开，恢复出完整的时间轴，以便送入时空融合算子（例如你的 YAML 里的 `transformer_layers: 4`）：

$$[B \times 16, 768] \xrightarrow{\text{View / Reshape}} [B, 16, 768]$$



*(如果使用 PyTorch 经典的 `nn.TransformerEncoder`，可能还需要重排为 `[16, B, 768]`，即 `[Seq_Len, Batch, Embed_Dim]`)*
* **第三步：时序交互（Temporal Fusion）**
通过时间步上的 Transformer Block 或前向时序算子融合多帧关联：

$$[B, 16, 768] \longrightarrow \text{Transformer Block (4 Layers)} \longrightarrow [B, 16, 768] \xrightarrow{\text{Mean / Layer Pooling}} \text{视频全局向量: } [B, 768]$$


* **向量空间映射逻辑**：
从**帧独立空间特征空间** $\longrightarrow$ **时序上下文关联空间** $\longrightarrow$ **全局视频质量表征空间**。
* 通过时间步上的 Transformer，网络开始建立帧与帧之间的前后文联系（即**时域特征**，如画面抖动、抽搐、编码卡顿等）。
* 最后的 `Mean / Layer Pooling`（通常对 16 帧的特征取平均，或者取最后一帧的 Token）将整个时序轴的信息坍缩，融合成一个代表整段 10 秒视频全局运动与画面质量的 $768$ 维综合向量。



#### 3. 回归层阶段 (Regression Head)

* **Tensor 形状变换**：
与 IQA 完全一致，消费融合了时空特征的视频全局向量：

$$[B, 768] \xrightarrow{\text{Linear(768, 512)}} [B, 512] \xrightarrow{\text{Linear(512, 256)}} [B, 256] \xrightarrow{\text{Linear(256, 1)}} \text{视频预测 MOS: } [B, 1]$$


* **向量空间映射逻辑**：
将同时包含“画面物理损伤（空间）”与“播放流畅度/时序运动失真（时间）”的**高级时空复合表征空间**，最终收敛映射到**一维连续视频主观质量空间（视频 MOS 标量空间）**。

---

### 💡 核心总结思维导图

| 阶段 | 纯图像 (IQA) Tensor 变换 | 视频流 (VQA) Tensor 变换 | 向量空间映射本质 |
| --- | --- | --- | --- |
| **1. 特征提取** | $[B, 3, 224, 224] \rightarrow [B, 7, 7, 768]$ | $[B \times 16, 3, 224, 224] \rightarrow [B \times 16, 7, 7, 768]$ | 低阶物理像素空间 $\rightarrow$ 高阶局部失真语义空间 |
| **2. 时空融合** | $[B, 7, 7, 768] \rightarrow [B, 768]$ | $[B \times 16, 7, 7, 768] \rightarrow [B, 16, 768] \rightarrow [B, 768]$ | 局部空间/时序分布式空间 $\rightarrow$ 全局整片/整段表征空间 |
| **3. 回归输出** | $[B, 768] \rightarrow [B, 512] \rightarrow [B, 1]$ | $[B, 768] \rightarrow [B, 512] \rightarrow [B, 1]$ | 高维抽象特征空间 $\rightarrow$ 一维人类主观感知单调打分空间（MOS） |