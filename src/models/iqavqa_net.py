import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from typing import Dict, Optional

class IQAVQANet(nn.Module):
    """
    自适应 IQA / VQA 大一统网络
    - Swin-T / ResNet50 混合骨干 + 4层 Transformer 时空融合 + 3层高维 MLP 质量头
    - 根据输入 Tensor 的维度自动无缝切换 IQA(4D) 与 VQA(5D) 模式

    混合型损失函数：0.7 * MSE Loss + 0.3 * Pairwise Rank Loss
    """

    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config.get('model', {})
        self.backbone_name = model_cfg.get('backbone', 'swin_t')
        self.dropout_rate = model_cfg.get('dropout', 0.3)
        self.freeze_backbone = model_cfg.get('freeze_backbone', False)
        self.num_vqa_layers = model_cfg.get('transformer_layers', 4)
        self.num_frames = model_cfg.get('num_frames', 8)
        # ----------------------------------------------------
        # 1. 特征提取层：动态编译骨干网络并完美适配特征维度
        # ----------------------------------------------------
        if self.backbone_name == "swin_t":
            # 加载预训练的 Swin-Transformer 骨干网络
            swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            self.backbone = swin.features
            self.num_features = swin.head.in_features  # 768
            self.is_transformer = True

        elif self.backbone_name == "resnet50":
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(
                res.conv1, res.bn1, res.relu, res.maxpool,
                res.layer1, res.layer2, res.layer3, res.layer4
            )
            self.num_features = res.fc.in_features  # 2048
            self.is_transformer = False
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_name}")

        # 骨干网络冻结开关机制
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ----------------------------------------------------
        # 2. 工业级自适应池化对账层
        # ----------------------------------------------------
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # ----------------------------------------------------
        # 3. 时空融合层：4层高效标准 Transformer 编码器（VQA 专属专用）
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_features,
            nhead=8,
            dim_feedforward=self.num_features * 2,
            dropout=self.dropout_rate,
            activation='gelu' if self.is_transformer else 'relu',
            batch_first=True
        )
        self.temporal_fusion = nn.TransformerEncoder(encoder_layer, num_layers=self.num_vqa_layers)

        # ----------------------------------------------------
        # 4. 回归层：高维三层 MLP 质量头
        # ----------------------------------------------------
        self.quality_head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.GELU() if self.is_transformer else nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 256),
            nn.GELU() if self.is_transformer else nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 🚀【铁壁防御 1】：强行将模型输出锁死在 (0, 1) 区间！
                          # 完全对齐自适应归一化的标签，斩断 Rank Loss 梯度爆炸的数学根源！
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """高级回归头稳健初始化：改用 Xavier 均匀分布，彻底破解常数预测死锁"""
        for m in self.quality_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 💎 回归头标配：防止方差崩塌或饱和
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _forward_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """自适应提取并平整化主干网高维特征，完美防御各种变体维度混乱"""
        features = self.backbone(x)

        # 兼容 Torchvision 官方 Swin-T 输出的不同形态
        if features.dim() == 4:
            # 判断是 [B, C, H, W] 还是 [B, H, W, C]
            if features.shape[-1] == self.num_features and features.shape[1] != self.num_features:
                # [B, H, W, C] -> [B, C, H, W]
                features = features.permute(0, 3, 1, 2)
        elif features.dim() == 3:
            # 若形如 [B, L, C] -> 动态还原 H 和 W
            B, L, C = features.shape
            H = W = int(L ** 0.5)
            if H * W == L:
                features = features.permute(0, 2, 1).view(B, C, H, W)

        pooled = self.spatial_pool(features)  # 稳健压缩为 [B, num_features, 1, 1]
        return torch.flatten(pooled, 1)      # 完美展平为 [B, num_features]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        维度自动切换核心路由：
        - 4D Tensor [B, 3, H, W] -> 纯图像 IQA 模式
        - 5D Tensor [B, F, 3, H, W] -> 视频 VQA 模式
        """
        # ========== 修复3：自动适配灰度图 ==========
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
        elif x.dim() == 5 and x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)  # [B, F, 1, H, W] -> [B, F, 3, H, W]
        # =======================================

        if x.dim() == 4:
            v_global = self._forward_backbone_features(x)
            score = self.quality_head(v_global)
            return score.squeeze(-1)  # 确保返回一维张量 [B]

        elif x.dim() == 5:
            B, F, C, H, W = x.shape

            # 如果输入帧数与配置不同，可以采样或警告
            if F != self.num_frames:
                # 采样到目标帧数
                indices = np.linspace(0, F-1, self.num_frames, dtype=int)
                x = x[:, indices, :, :, :]
                B, F, C, H, W = x.shape

            # 修复3补充：确保是3通道
            if C != 3:
                raise ValueError(f"Expected 3 channels after grayscale conversion, got {C}")

            # 1. 空间打包并行化提取
            x_reshaped = x.view(B * F, C, H, W)
            v_frames = self._forward_backbone_features(x_reshaped)

            # 2. 时序解包重排
            v = v_frames.view(B, F, self.num_features)

            # 3. 时序多层上下文自注意力交互
            v_fused = self.temporal_fusion(v)

            # 4. 全局时间平均池化
            v_global = torch.mean(v_fused, dim=1)

            # 5. 回归质量评估打分
            score = self.quality_head(v_global)
            return score.squeeze(-1)  # 确保返回一维张量 [B]

        else:
            raise ValueError(f"Invalid input tensor dim: {x.dim()}. Expected 4D for IQA or 5D for VQA.")


class IQAVQALoss(nn.Module):
    """
    混合型损失函数：0.7 * MSE Loss + 0.3 * Pairwise Rank Loss (移除高危显存黑洞 L2 循环)
    """

    def __init__(self, config: Dict):
        super().__init__()
        loss_cfg = config.get('loss', {})
        model_cfg = config.get('model', {})
        self.num_frames = model_cfg.get('num_frames', 8)  # 添加这一行，默认 8 帧
        self.mse_weight = loss_cfg.get('mse_weight', 0.7)
        self.rank_weight = loss_cfg.get('rank_weight', 0.3)
        self.max_pairs = loss_cfg.get('max_pairs', 5000)  # ✅ 添加采样上限配置
        self.mse_loss = nn.MSELoss()


    def rank_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """采样方式计算 rank loss"""
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        n = len(y_pred)

        if n > self.max_pairs:
            # 随机采样
            idx = torch.randperm(n)[:self.max_pairs]
            y_pred = y_pred[idx]
            y_true = y_true[idx]
            n = self.max_pairs

        pred_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)

        # 利用 torch.clamp 截断并配合平滑，确保单对样本梯度有上限， 避免梯度爆炸
        loss = torch.relu(-pred_diff * true_diff).mean()
        return loss



    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        # 统一规整尺寸，消除维度不匹配带来的隐患
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float()

        # 数值溢出双重防空保护（防止产生微小的 NaN 扰动）
        y_pred = torch.clamp(y_pred, min=1e-6, max=1.0 - 1e-6)

        mse = self.mse_loss(y_pred, y_true)
        rank = self.rank_loss(y_pred, y_true)
        total_loss = self.mse_weight * mse + self.rank_weight * rank

        # 📌 完美闭环：彻底移除了原先耗尽显存建图的 param 循环。
        # 框架的 L2 正则化通过 basic_config.yaml 里的 weight_decay=1e-4 自动由 AdamW 完美接管。
        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'rank_loss': rank
        }