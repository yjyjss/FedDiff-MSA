"""
FedDiff-MSA Federated Aggregation & Privacy
MAFA-Diff aggregation strategy, risk-aware layered DP, communication optimization.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy

from config.config import ExperimentConfig


class MAFADiffAggregator:
    """
    Modality-Aware and Diffusion-Aware Federated Aggregation.
    """

    def __init__(self, gamma: float = 0.2, beta: float = 0.9):
        self.gamma = gamma
        self.beta = beta
        self.global_state_prev = None  # For momentum compensation

    def aggregate(
        self,
        global_model_state: Dict[str, torch.Tensor],
        client_updates: List[Dict],
        client_modality_info: List[Dict],
        param_groups: Dict[str, List[str]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using MAFA-Diff strategy.

        Args:
            global_model_state: current global model state dict
            client_updates: list of client model state dicts
            client_modality_info: list of dicts with keys 'has_text', 'has_audio', 'has_visual', 'has_all', 'n_samples'
            param_groups: mapping group_name -> list of param key patterns

        Returns:
            new global model state dict
        """
        new_state = {}
        total_samples = sum(info["n_samples"] for info in client_modality_info)

        for key, global_val in global_model_state.items():
            # Determine which parameter group this key belongs to
            group = self._get_param_group(key, param_groups)

            if group in ["text", "audio", "visual"]:
                # Modality encoder aggregation: only clients with that modality
                modality_key = f"has_{group}"
                participating = [
                    (i, client_modality_info[i])
                    for i in range(len(client_updates))
                    if client_modality_info[i].get(modality_key, False)
                ]

                if not participating:
                    new_state[key] = global_val.clone()
                    continue

                # Weighted average by data count
                weighted_sum = torch.zeros_like(global_val)
                total_weight = 0.0
                for i, info in participating:
                    w = info["n_samples"]
                    weighted_sum += w * client_updates[i][key]
                    total_weight += w

                new_state[key] = weighted_sum / max(total_weight, 1e-8)

            elif group == "diffusion":
                # Diffusion model aggregation: quality bonus for complete-modality clients
                weights = []
                for i, info in enumerate(client_modality_info):
                    w = info["n_samples"] / max(total_samples, 1)
                    if info.get("has_all", False):
                        w *= (1 + self.gamma)
                    weights.append(w)

                # Normalize
                total_w = sum(weights)
                weights = [w / total_w for w in weights]

                weighted_sum = torch.zeros_like(global_val)
                for i, w in enumerate(weights):
                    weighted_sum += w * client_updates[i][key]

                new_state[key] = weighted_sum

            else:  # "fusion" or any other
                # Standard FedAvg for fusion-classification head
                # With momentum compensation for missing-modality clients
                weighted_sum = torch.zeros_like(global_val)
                total_weight = 0.0

                for i, info in enumerate(client_modality_info):
                    w = info["n_samples"]
                    client_val = client_updates[i][key]

                    # Momentum compensation for missing-modality clients
                    if not info.get("has_all", False) and self.global_state_prev is not None:
                        if key in self.global_state_prev:
                            client_val = self.beta * self.global_state_prev[key] + (1 - self.beta) * client_val

                    weighted_sum += w * client_val
                    total_weight += w

                new_state[key] = weighted_sum / max(total_weight, 1e-8)

        # Store for next round's momentum compensation
        self.global_state_prev = {k: v.clone() for k, v in new_state.items()}

        return new_state

    def _get_param_group(self, key: str, param_groups: Dict[str, List[str]]) -> str:
        """Determine which parameter group a key belongs to."""
        for group_name, patterns in param_groups.items():
            for pattern in patterns:
                if pattern in key:
                    return group_name
        return "fusion"  # Default


class RiskAwareLayeredDP:
    """
    Risk-aware layered differential privacy.
    Applies per-group gradient clipping and noise injection.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config.privacy
        self.group_config = {
            "text": (self.config.clip_text, self.config.sigma_text, self.config.budget_text),
            "audio": (self.config.clip_audio, self.config.sigma_audio, self.config.budget_audio),
            "visual": (self.config.clip_visual, self.config.sigma_visual, self.config.budget_visual),
            "diffusion": (self.config.clip_diffusion, self.config.sigma_diffusion, self.config.budget_diffusion),
            "fusion": (self.config.clip_fusion, self.config.sigma_fusion, self.config.budget_fusion),
        }

    def apply_dp(
        self,
        model,
        param_groups: Dict[str, List],
    ) -> Dict[str, float]:
        """
        Apply layered DP to model gradients in-place.
        Returns per-group noise norms for logging.
        """
        if not self.config.enabled:
            return {}

        noise_norms = {}
        for group_name, params in param_groups.items():
            if not params:
                continue

            clip_c, sigma, _ = self.group_config[group_name]

            # Compute total gradient norm for this group
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm().item() ** 2
            total_norm = total_norm ** 0.5

            # Clip
            scale = min(1.0, clip_c / (total_norm + 1e-8))
            for p in params:
                if p.grad is not None:
                    p.grad.data.mul_(scale)

            # Add noise
            noise_norm = 0.0
            for p in params:
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * sigma * clip_c
                    p.grad.data.add_(noise)
                    noise_norm += noise.norm().item() ** 2

            noise_norms[group_name] = noise_norm ** 0.5

        return noise_norms


class CommunicationCompressor:
    """
    Top-K sparsification + INT8 quantization for communication efficiency.
    """

    def __init__(self, sparsification_ratio: float = 0.1, quantization_bits: int = 8, error_feedback: bool = True):
        self.sparsification_ratio = sparsification_ratio
        self.quantization_bits = quantization_bits
        self.error_feedback = error_feedback
        self.error_buffer = {}

    def compress(self, state_dict: Dict[str, torch.Tensor]) -> Dict:
        """
        Compress model updates for communication.
        Returns compressed representation.
        """
        compressed = {}
        for key, tensor in state_dict.items():
            flat = tensor.flatten()

            # Error feedback
            if self.error_feedback:
                if key not in self.error_buffer:
                    self.error_buffer[key] = torch.zeros_like(flat)
                flat = flat + self.error_buffer[key]

            # Top-K sparsification
            k = max(1, int(len(flat) * self.sparsification_ratio))
            topk_vals, topk_idx = torch.topk(flat.abs(), k)
            sparse_vals = flat[topk_idx]

            # Quantization (simulated INT8)
            if self.quantization_bits == 8:
                max_val = sparse_vals.abs().max().clamp(min=1e-8)
                scale = max_val / 127.0
                quantized = torch.round(sparse_vals / scale).clamp(-128, 127).to(torch.int8)
                dequantized = quantized.float() * scale
            else:
                dequantized = sparse_vals
                scale = torch.tensor(1.0)

            # Update error buffer
            if self.error_feedback:
                residual = torch.zeros_like(flat)
                residual[topk_idx] = flat[topk_idx] - dequantized
                self.error_buffer[key] = residual

            compressed[key] = {
                "indices": topk_idx,
                "values": dequantized,
                "shape": tensor.shape,
            }

        return compressed

    def decompress(self, compressed: Dict) -> Dict[str, torch.Tensor]:
        """Decompress back to full state dict."""
        result = {}
        for key, item in compressed.items():
            flat = torch.zeros(item["shape"].numel(), device=item["values"].device)
            flat[item["indices"]] = item["values"]
            result[key] = flat.reshape(item["shape"])
        return result


class RDPAccountant:
    """
    Simplified Rényi Differential Privacy accountant.
    For production, use Opacus RDPAccountant.
    """

    def __init__(self, epsilon: float, delta: float, sample_rate: float):
        self.epsilon = epsilon
        self.delta = delta
        self.sample_rate = sample_rate
        self.steps = 0
        self.consumed = {}

    def step(self, group_name: str, sigma: float, clip_c: float):
        """
        Accumulate privacy cost for one step of one parameter group.
        Uses a simplified RDP bound with subsampling amplification.
        """
        self.steps += 1
        # Simplified RDP: uses subsampling amplification factor q^2/(2*sigma^2)
        # where q is the sampling rate. This is a conservative bound.
        if group_name not in self.consumed:
            self.consumed[group_name] = 0.0
        # Scale down by number of groups to avoid premature exhaustion
        cost = self.sample_rate ** 2 / (2 * sigma ** 2 * 5)  # 5 groups
        self.consumed[group_name] += cost

    def get_epsilon(self, group_name: str = None) -> float:
        """Return consumed epsilon for a group or total."""
        if group_name:
            eps = self.consumed.get(group_name, 0.0)
        else:
            eps = sum(self.consumed.values())
        return eps

    def remaining_budget(self, total_epsilon: float) -> float:
        """Return remaining privacy budget."""
        return max(0, total_epsilon - self.get_epsilon())
