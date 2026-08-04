"""
FedDiff-MSA Diffusion Module
Conditional diffusion model for modality recovery in 256-dim feature space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) integer or float timesteps
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (batch, dim)


class ResBlock(nn.Module):
    """Residual block with time/condition injection."""

    def __init__(self, in_dim: int, out_dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning."""

    def __init__(self, dim: int, cond_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            batch_first=True, dropout=dropout,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim) -> (batch, 1, dim)
        # cond: (batch, cond_dim) -> (batch, 1, cond_dim)
        x_2d = x.unsqueeze(1)
        cond_2d = cond.unsqueeze(1)

        h = self.norm(x_2d)
        attn_out, _ = self.attn(h, cond_2d, cond_2d)
        x_2d = x_2d + attn_out

        h2 = self.norm_ff(x_2d)
        ff_out = self.ff(h2)
        x_2d = x_2d + ff_out

        return x_2d.squeeze(1)


class ConditionalUNet(nn.Module):
    """
    Lightweight conditional U-Net for feature-space denoising.
    Operates on (batch, hidden_dim) features.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        time_dim: int = 128,
        cond_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.cond_proj = nn.Linear(cond_dim, time_dim)

        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_d = feature_dim if i == 0 else feature_dim
            self.enc_blocks.append(ResBlock(in_d, feature_dim, time_dim, dropout))
            self.enc_blocks.append(CrossAttentionBlock(feature_dim, cond_dim, num_heads, dropout))

        # Bottleneck
        self.bottleneck = ResBlock(feature_dim, feature_dim, time_dim, dropout)
        self.bottleneck_attn = CrossAttentionBlock(feature_dim, cond_dim, num_heads, dropout)

        # Decoder blocks (with skip connections)
        self.dec_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.dec_blocks.append(ResBlock(feature_dim * 2, feature_dim, time_dim, dropout))
            self.dec_blocks.append(CrossAttentionBlock(feature_dim, cond_dim, num_heads, dropout))

        # Output
        self.out_norm = nn.LayerNorm(feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy features (batch, feature_dim)
            t: timesteps (batch,)
            cond: condition features (batch, cond_dim)
        Returns:
            predicted noise (batch, feature_dim)
        """
        t_emb = self.time_embed(t)  # (batch, time_dim)
        t_emb = t_emb + self.cond_proj(cond)  # Fuse time and condition

        # Encoder
        skips = []
        for i in range(0, len(self.enc_blocks), 2):
            x = self.enc_blocks[i](x, t_emb)
            x = self.enc_blocks[i + 1](x, cond)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb)
        x = self.bottleneck_attn(x, cond)

        # Decoder
        for i in range(0, len(self.dec_blocks), 2):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=-1)
            x = self.dec_blocks[i](x, t_emb)
            x = self.dec_blocks[i + 1](x, cond)

        x = self.out_norm(x)
        return self.out_proj(x)


class CosineNoiseScheduler:
    """Cosine noise schedule for diffusion."""

    def __init__(self, num_steps: int = 50, s: float = 0.008):
        self.num_steps = num_steps
        self.s = s

        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        self.alphas_cumprod = alphas_cumprod
        self.alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.betas = 1.0 - self.alphas
        self.betas = torch.clamp(self.betas, 1e-4, 0.02)

        # Recompute
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        return self

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at timestep t."""
        noise = torch.randn_like(x_0)
        sqrt_acp = torch.sqrt(self.alphas_cumprod[t]).unsqueeze(-1)
        sqrt_omacp = torch.sqrt(1 - self.alphas_cumprod[t]).unsqueeze(-1)
        x_t = sqrt_acp * x_0 + sqrt_omacp * noise
        return x_t, noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device)


class DiffusionModel(nn.Module):
    """
    Complete diffusion model: U-Net + scheduler + forward/reverse process.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        time_dim: int = 128,
        cond_dim: int = 256,
        num_steps: int = 50,
        num_layers: int = 3,
        num_heads: int = 2,
        dropout: float = 0.1,
        eta: float = 0.0,
        tbtt_window: int = 10,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.eta = eta
        self.tbtt_window = tbtt_window
        self.feature_dim = feature_dim

        self.unet = ConditionalUNet(
            feature_dim, time_dim, cond_dim, num_layers, num_heads, dropout
        )
        self.scheduler = CosineNoiseScheduler(num_steps)

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def forward_diffuse(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean features."""
        return self.scheduler.add_noise(x_0, t)

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise using U-Net."""
        return self.unet(x_t, t, cond)

    @torch.no_grad()
    def sample_ddim(
        self,
        cond: torch.Tensor,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        DDIM sampling for modality recovery.
        Returns recovered features (batch, feature_dim).
        """
        x = torch.randn(batch_size, self.feature_dim, device=device)

        # DDIM: use a subset of timesteps
        step_indices = list(range(0, self.num_steps, max(1, self.num_steps // 10)))
        step_indices = step_indices[::-1]  # Reverse

        for i, t_idx in enumerate(step_indices):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

            pred_noise = self.unet(x, t, cond)

            alpha_t = self.scheduler.alphas_cumprod[t_idx]
            prev_idx = step_indices[i + 1] if i + 1 < len(step_indices) else -1
            alpha_prev = self.scheduler.alphas_cumprod[prev_idx] if prev_idx >= 0 else torch.tensor(1.0, device=device)

            # DDIM update
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            sigma = self.eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t + 1e-8) * (1 - alpha_t / (alpha_prev + 1e-8))
            )

            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * pred_noise
            noise = sigma * torch.randn_like(x) if self.eta > 0 else 0

            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + noise

        return x

    def sample_for_training(self, cond: torch.Tensor, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Sample for training (with gradient through last tbtt_window steps).
        Uses truncated backpropagation through time.
        """
        x = torch.randn(batch_size, self.feature_dim, device=device)

        step_indices = list(range(0, self.num_steps, max(1, self.num_steps // 10)))
        step_indices = step_indices[::-1]

        total_steps = len(step_indices)
        # Detach early steps, keep gradient for last tbtt_window
        cutoff = max(0, total_steps - self.tbtt_window)

        for i, t_idx in enumerate(step_indices):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

            if i == cutoff:
                x = x.detach()  # Truncate gradient here

            pred_noise = self.unet(x, t, cond)

            alpha_t = self.scheduler.alphas_cumprod[t_idx]
            prev_idx = step_indices[i + 1] if i + 1 < len(step_indices) else -1
            alpha_prev = self.scheduler.alphas_cumprod[prev_idx] if prev_idx >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            sigma = 0.0  # Deterministic during training
            dir_xt = torch.sqrt(1 - alpha_prev) * pred_noise
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt

        return x
