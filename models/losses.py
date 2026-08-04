"""
FedDiff-MSA Loss Functions
Triple loss: denoising + emotion consistency + cross-modal contrastive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DenoisingLoss(nn.Module):
    """MSE loss between predicted and actual noise."""

    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predicted_noise, target_noise)


class EmotionConsistencyLoss(nn.Module):
    """Cross-entropy loss for emotion classification (computed on recovered representations)."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class CrossModalContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss.
    Pulls recovered representations close to real representations.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        recovered: torch.Tensor,   # (batch, dim)
        real: torch.Tensor,         # (batch, dim)
    ) -> torch.Tensor:
        if recovered.size(0) < 2:
            return torch.tensor(0.0, device=recovered.device, requires_grad=True)

        # Normalize
        rec_norm = F.normalize(recovered, dim=-1)
        real_norm = F.normalize(real, dim=-1)

        # Similarity matrix
        sim = torch.matmul(rec_norm, real_norm.T) / self.temperature  # (batch, batch)

        # Positive pairs on diagonal
        labels = torch.arange(recovered.size(0), device=recovered.device)
        loss = F.cross_entropy(sim, labels)

        return loss


class ConfidenceEvaluatorLoss(nn.Module):
    """L1 loss for confidence evaluator training."""

    def forward(self, predicted_conf: torch.Tensor, target_conf: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(predicted_conf, target_conf)


class TripleLoss(nn.Module):
    """
    Combined loss: L_total = L_cls + lambda_1 * L_diff + lambda_2 * L_emo + lambda_3 * L_contrast.
    """

    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_emo: float = 0.5,
        lambda_contrast: float = 0.1,
        lambda_conf: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_emo = lambda_emo
        self.lambda_contrast = lambda_contrast
        self.lambda_conf = lambda_conf

        self.denoising_loss = DenoisingLoss()
        self.emotion_loss = EmotionConsistencyLoss()
        self.contrastive_loss = CrossModalContrastiveLoss(temperature)
        self.conf_loss = ConfidenceEvaluatorLoss()

    def forward(
        self,
        # Classification
        logits: torch.Tensor,
        labels: torch.Tensor,
        # Denoising (only for available modality samples doing virtual missing training)
        pred_noise: Optional[torch.Tensor] = None,
        target_noise: Optional[torch.Tensor] = None,
        # Emotion consistency (only for missing modality samples)
        emo_logits: Optional[torch.Tensor] = None,
        emo_labels: Optional[torch.Tensor] = None,
        # Contrastive (only for available modality samples)
        recovered_feat: Optional[torch.Tensor] = None,
        real_feat: Optional[torch.Tensor] = None,
        # Confidence evaluator
        pred_conf: Optional[torch.Tensor] = None,
        target_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all applicable losses and return a dict.
        """
        losses = {}
        total = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # L_cls: always computed
        l_cls = self.emotion_loss(logits, labels)
        losses["cls"] = l_cls
        total = total + l_cls

        # L_diff: denoising loss (available modality samples, virtual missing training)
        if pred_noise is not None and target_noise is not None:
            l_diff = self.denoising_loss(pred_noise, target_noise)
            losses["diff"] = l_diff
            total = total + self.lambda_diff * l_diff

        # L_emo: emotion consistency loss (missing modality samples)
        if emo_logits is not None and emo_labels is not None:
            l_emo = self.emotion_loss(emo_logits, emo_labels)
            losses["emo"] = l_emo
            total = total + self.lambda_emo * l_emo

        # L_contrast: contrastive loss (available modality samples)
        if recovered_feat is not None and real_feat is not None:
            l_contrast = self.contrastive_loss(recovered_feat, real_feat)
            losses["contrast"] = l_contrast
            total = total + self.lambda_contrast * l_contrast

        # L_conf: confidence evaluator loss
        if pred_conf is not None and target_conf is not None:
            l_conf = self.conf_loss(pred_conf, target_conf)
            losses["conf"] = l_conf
            total = total + self.lambda_conf * l_conf

        losses["total"] = total
        return losses
