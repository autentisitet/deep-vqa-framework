# src/models/losses.py
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.config.schemas import Config


class IQAVQALoss(nn.Module):
    """
    Loss function with task-aware weighting.

    IQA: MSE + Rank Loss
    VQA: MSE + Rank Loss + PLCC Loss
    """

    def __init__(self, cfg: Config):
        super().__init__()
        loss_cfg = cfg.loss

        self.iqa_mse_weight = loss_cfg.iqa_mse_weight if loss_cfg.iqa_mse_weight is not None else loss_cfg.mse_weight
        self.iqa_rank_weight = loss_cfg.iqa_rank_weight if loss_cfg.iqa_rank_weight is not None else loss_cfg.rank_weight
        self.iqa_plcc_weight = loss_cfg.iqa_plcc_weight if loss_cfg.iqa_plcc_weight is not None else 0.0

        self.vqa_mse_weight = loss_cfg.vqa_mse_weight if loss_cfg.vqa_mse_weight is not None else loss_cfg.mse_weight
        self.vqa_rank_weight = loss_cfg.vqa_rank_weight if loss_cfg.vqa_rank_weight is not None else loss_cfg.rank_weight
        self.vqa_plcc_weight = loss_cfg.vqa_plcc_weight if loss_cfg.vqa_plcc_weight is not None else loss_cfg.plcc_weight

        self.max_pairs = loss_cfg.max_pairs
        self.mse_loss = nn.MSELoss()

    def plcc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - PLCC."""
        pred = pred.view(-1)
        target = target.view(-1)

        pred_n = pred - pred.mean()
        target_n = target - target.mean()

        plcc = (pred_n * target_n).sum() / (
            torch.sqrt((pred_n**2).sum()) * torch.sqrt((target_n**2).sum()) + 1e-8
        )
        return 1 - plcc

    def rank_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pairwise rank loss with sampling."""
        pred = pred.view(-1)
        target = target.view(-1)
        n = len(pred)

        if n > self.max_pairs:
            idx = torch.randperm(n)[:self.max_pairs]
            pred = pred[idx]
            target = target[idx]
            n = self.max_pairs

        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1)

        loss = torch.relu(-pred_diff * target_diff)
        return loss.mean() + 1e-8

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: Optional[nn.Module] = None,
        mode: str = "iqa",
    ) -> Dict[str, torch.Tensor]:
        pred = pred.view(-1)
        target = target.view(-1).float()
        pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)

        mse = self.mse_loss(pred, target)
        rank = self.rank_loss(pred, target)
        plcc = self.plcc_loss(pred, target)

        if mode == "iqa":
            total = self.iqa_mse_weight * mse + self.iqa_rank_weight * rank
        else:
            total = self.vqa_mse_weight * mse + self.vqa_rank_weight * rank + self.vqa_plcc_weight * plcc

        return {
            "total_loss": total,
            "mse_loss": mse,
            "rank_loss": rank,
            "plcc_loss": plcc,
            "mode": mode,
        }


__all__ = ["IQAVQALoss"]
