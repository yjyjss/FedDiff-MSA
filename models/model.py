"""
FedDiff-MSA Encoders and Classifier
Modality encoders, fusion module, confidence evaluator, and emotion classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class ModalityEncoder(nn.Module):
    """Single modality encoder: projects raw features to hidden_dim."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConfidenceEvaluator(nn.Module):
    """
    Lightweight MLP to score recovery quality.
    Input: concat(recovered_repr, condition) -> scalar in (0, 1).
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, recovered: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([recovered, cond], dim=-1)
        return self.net(x).squeeze(-1)


class ConfidenceAwareFusion(nn.Module):
    """
    Confidence-aware cross-modal attention fusion.
    Uses confidence scores to modulate attention to each modality.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head self-attention across modalities
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout,
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        modality_features: List[torch.Tensor],
        confidence_scores: List[float],
    ) -> torch.Tensor:
        """
        Args:
            modality_features: list of (batch, hidden_dim) tensors
            confidence_scores: list of float confidence per modality
        Returns:
            fused: (batch, hidden_dim)
        """
        # Stack into sequence: (batch, num_modalities, hidden_dim)
        stacked = torch.stack(modality_features, dim=1)

        # Build attention mask from confidence scores
        batch_size = stacked.size(0)
        num_modalities = len(modality_features)

        # Confidence-weighted attention bias
        conf_tensor = torch.tensor(
            confidence_scores, device=stacked.device, dtype=stacked.dtype
        )  # (num_modalities,)

        # Apply confidence as additive mask: low confidence -> large negative bias
        # Shape: (batch, num_modalities) -> for key_padding_mask
        # key_padding_mask: True means "ignore this position"
        conf_mask = conf_tensor.unsqueeze(0).expand(batch_size, -1)  # (batch, num_modalities)
        key_padding_mask = conf_mask < 0.1  # Ignore very low confidence modalities

        # Self-attention
        attn_out, _ = self.attn(
            stacked, stacked, stacked,
            key_padding_mask=key_padding_mask if key_padding_mask.any() else None,
        )

        # Apply confidence weighting
        conf_weight = conf_tensor.unsqueeze(0).unsqueeze(-1)  # (1, num_modalities, 1)
        attn_out = attn_out * conf_weight

        # Residual
        x = self.norm(stacked + attn_out)

        # FFN
        x = x + self.ffn(x)

        # Gating: compute per-modality gates
        gates = self.gate(x)  # (batch, num_modalities, 1)
        gates = F.softmax(gates, dim=1)

        # Weighted sum
        fused = (x * gates).sum(dim=1)  # (batch, hidden_dim)

        return fused


class EmotionClassifier(nn.Module):
    """
    Emotion classification head.
    Takes fused representation and outputs class logits.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 7,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused)


class FedDiffModel(nn.Module):
    """
    Complete FedDiff-MSA model: encoders + diffusion + fusion + classifier.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        m = config.model

        # Modality encoders
        self.text_encoder = ModalityEncoder(m.text_dim, m.hidden_dim, m.dropout)
        self.audio_encoder = ModalityEncoder(m.audio_dim, m.hidden_dim, m.dropout)
        self.visual_encoder = ModalityEncoder(m.visual_dim, m.hidden_dim, m.dropout)

        # Diffusion model (import here to avoid circular imports)
        from models.diffusion import DiffusionModel
        self.diffusion = DiffusionModel(
            feature_dim=m.hidden_dim,
            time_dim=m.time_embed_dim,
            cond_dim=m.hidden_dim,
            num_steps=m.diffusion_steps,
            num_layers=m.unet_layers,
            num_heads=m.unet_heads,
            dropout=m.dropout,
            eta=m.diffusion_eta,
            tbtt_window=m.tbtt_window,
        )

        # Confidence evaluator
        self.confidence_eval = ConfidenceEvaluator(m.hidden_dim, m.conf_hidden)

        # Fusion
        self.fusion = ConfidenceAwareFusion(m.hidden_dim, m.num_heads, m.dropout)

        # Classifier
        self.classifier = EmotionClassifier(m.hidden_dim, m.num_classes, m.classifier_hidden, m.dropout)

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Return parameter groups for federated aggregation and DP."""
        return {
            "text": list(self.text_encoder.parameters()),
            "audio": list(self.audio_encoder.parameters()),
            "visual": list(self.visual_encoder.parameters()),
            "diffusion": list(self.diffusion.parameters()),
            "fusion": list(self.fusion.parameters()) + list(self.classifier.parameters()) + list(self.confidence_eval.parameters()),
        }

    def get_all_parameters(self) -> List[nn.Parameter]:
        """Return all trainable parameters."""
        groups = self.get_param_groups()
        params = []
        for g in groups.values():
            params.extend(g)
        return params

    def encode_modalities(self, batch: Dict) -> Dict[str, Optional[torch.Tensor]]:
        """Encode available modalities to hidden_dim representations."""
        device = next(self.parameters()).device
        mask = batch["modality_mask"]  # (batch, 3)

        encoded = {}
        # Text
        if batch.get("text") is not None:
            encoded["text"] = self.text_encoder(batch["text"].to(device))
        else:
            encoded["text"] = None

        # Audio
        if batch.get("audio") is not None:
            encoded["audio"] = self.audio_encoder(batch["audio"].to(device))
        else:
            encoded["audio"] = None

        # Visual
        if batch.get("visual") is not None:
            encoded["visual"] = self.visual_encoder(batch["visual"].to(device))
        else:
            encoded["visual"] = None

        return encoded

    def build_condition(self, encoded: Dict) -> torch.Tensor:
        """Build condition vector from available modalities (mean pooling)."""
        avail = [v for v in encoded.values() if v is not None]
        if not avail:
            # Fallback: zero condition
            batch_size = 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.config.model.hidden_dim, device=device)

        # Average all available modality features
        cond = torch.stack(avail, dim=0).mean(dim=0)
        return cond

    def identify_missing_modalities(self, mask: torch.Tensor) -> List[str]:
        """Return list of missing modality names for the batch."""
        # mask: (batch, 3) -> check first sample (assume consistent within batch)
        m = mask[0]
        missing = []
        if m[0] < 0.5:
            missing.append("text")
        if m[1] < 0.5:
            missing.append("audio")
        if m[2] < 0.5:
            missing.append("visual")
        return missing

    def forward(self, batch: Dict, training: bool = True) -> Dict:
        """
        Full forward pass: encode -> recover missing -> fuse -> classify.

        Returns dict with keys: logits, recovered_features, condition, confidence_scores
        """
        device = next(self.parameters()).device
        mask = batch["modality_mask"]

        # Step 1: Encode available modalities
        encoded = self.encode_modalities(batch)

        # Step 2: Identify missing modalities
        missing = self.identify_missing_modalities(mask)

        # Step 3: Build condition from available modalities
        cond = self.build_condition(encoded)

        # Step 4: Recover missing modalities via diffusion
        batch_size = mask.size(0)
        recovered = {}
        confidence_scores = {}

        for mod_name in missing:
            if training:
                recovered_feat = self.diffusion.sample_for_training(
                    cond, batch_size=batch_size, device=device
                )
            else:
                recovered_feat = self.diffusion.sample_ddim(
                    cond, batch_size=batch_size, device=device
                )
            recovered[mod_name] = recovered_feat
            conf = self.confidence_eval(recovered_feat, cond)
            confidence_scores[mod_name] = conf

        # Step 5: Assemble modality features list and confidence list
        modality_order = ["text", "audio", "visual"]
        modality_features = []
        conf_list = []

        for mod in modality_order:
            if encoded[mod] is not None:
                modality_features.append(encoded[mod])
                conf_list.append(1.0)
            elif mod in recovered:
                modality_features.append(recovered[mod])
                conf_list.append(confidence_scores[mod].mean().item())
            else:
                # Should not happen, but handle gracefully
                zeros = torch.zeros(batch_size, self.config.model.hidden_dim, device=device)
                modality_features.append(zeros)
                conf_list.append(0.0)

        # Step 6: Fuse
        fused = self.fusion(modality_features, conf_list)

        # Step 7: Classify
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "fused": fused,
            "recovered": recovered,
            "condition": cond,
            "confidence_scores": confidence_scores,
            "encoded": encoded,
        }
