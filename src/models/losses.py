# src/models/losses.py
from typing import Dict

import torch
import torch.nn as nn

from src.config.schemas import Config


class IQAVQALoss(nn.Module):
    """
    Hybrid quality-assessment loss.

    SmoothL1 provides robust MOS regression, while pairwise logistic ranking
    directly optimizes quality ordering for both IQA and VQA.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        loss_cfg = cfg.loss

        self.smooth_l1_weight = loss_cfg.smooth_l1_weight
        self.rank_weight = loss_cfg.rank_weight
        self.max_pairs = loss_cfg.max_pairs
        self.rank_epsilon = loss_cfg.rank_epsilon
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=loss_cfg.huber_delta)

    def rank_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pairwise logistic rank loss, ignoring indistinguishable MOS pairs."""
        pred = pred.view(-1)
        target = target.view(-1)
        n = len(pred)
        if n < 2:
            return pred.sum() * 0.0

        pair_i, pair_j = torch.triu_indices(n, n, offset=1, device=pred.device)
        target_diff = target[pair_i] - target[pair_j]
        valid = target_diff.abs() > self.rank_epsilon
        pair_i, pair_j, target_diff = pair_i[valid], pair_j[valid], target_diff[valid]
        if pair_i.numel() == 0:
            return pred.sum() * 0.0

        if pair_i.numel() > self.max_pairs:
            selected = torch.randperm(pair_i.numel(), device=pred.device)[: self.max_pairs]
            pair_i, pair_j, target_diff = pair_i[selected], pair_j[selected], target_diff[selected]

        pred_diff = pred[pair_i] - pred[pair_j]
        target_sign = torch.sign(target_diff)
        pair_weight = target_diff.abs().clamp(max=1.0)
        return (pair_weight * torch.nn.functional.softplus(-target_sign * pred_diff)).sum() / pair_weight.sum()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = pred.view(-1)
        target = target.view(-1).float()
        pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)

        smooth_l1 = self.smooth_l1_loss(pred, target)
        rank = self.rank_loss(pred, target)

        total = self.smooth_l1_weight * smooth_l1 + self.rank_weight * rank

        return {
            "total_loss": total,
            "smooth_l1_loss": smooth_l1,
            "rank_loss": rank,
        }


__all__ = ["IQAVQALoss"]
