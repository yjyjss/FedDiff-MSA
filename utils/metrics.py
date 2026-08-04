"""
FedDiff-MSA Evaluation Utilities
Metrics computation and experiment-specific evaluation functions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 7,
) -> Dict:
    """Compute standard classification metrics."""
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "n_samples": len(labels),
    }


def compute_mmd(
    real_features: torch.Tensor,
    recovered_features: torch.Tensor,
    kernel: str = "rbf",
    gamma: float = None,
) -> float:
    """
    Maximum Mean Discrepancy between real and recovered features.
    Lower is better.
    """
    if gamma is None:
        gamma = 1.0 / real_features.size(-1)

    real = real_features.detach().cpu().numpy()
    recovered = recovered_features.detach().cpu().numpy()

    from sklearn.metrics.pairwise import rbf_kernel

    K_rr = rbf_kernel(real, real, gamma=gamma)
    K_pp = rbf_kernel(recovered, recovered, gamma=gamma)
    K_rp = rbf_kernel(real, recovered, gamma=gamma)

    mmd = K_rr.mean() + K_pp.mean() - 2 * K_rp.mean()
    return float(max(mmd, 0.0))


def compute_cosine_similarity(
    real_features: torch.Tensor,
    recovered_features: torch.Tensor,
) -> float:
    """Average cosine similarity between real and recovered features."""
    import torch.nn.functional as F
    sim = F.cosine_similarity(real_features, recovered_features, dim=-1)
    return float(sim.mean().item())


def compute_classification_consistency(
    real_logits: torch.Tensor,
    recovered_logits: torch.Tensor,
) -> float:
    """
    Classification consistency: fraction of samples where real and recovered
    features produce the same prediction.
    """
    real_preds = real_logits.argmax(dim=-1)
    recovered_preds = recovered_logits.argmax(dim=-1)
    return float((real_preds == recovered_preds).float().mean().item())


def evaluate_recovery_quality(
    model,
    dataloader,
    device: torch.device,
    modality_to_remove: str = "visual",
) -> Dict:
    """
    Evaluate diffusion recovery quality for a specific modality.
    Compares: zero-padding, mean-filling, autoencoder, diffusion.

    Returns dict with MMD, cosine sim, classification consistency for each method.
    """
    model.eval()
    results = {
        "zero_padding": {"mmd": 0.0, "cosine_sim": 0.0, "cls_consistency": 0.0},
        "mean_filling": {"mmd": 0.0, "cosine_sim": 0.0, "cls_consistency": 0.0},
        "diffusion": {"mmd": 0.0, "cosine_sim": 0.0, "cls_consistency": 0.0},
    }

    real_feats = []
    zero_pad_feats = []
    mean_fill_feats = []
    diffusion_feats = []
    real_logits_list = []
    diffusion_logits_list = []

    modality_idx = {"text": 0, "audio": 1, "visual": 2}
    remove_idx = modality_idx[modality_to_remove]

    with torch.no_grad():
        for batch in dataloader:
            # Get real features for the target modality
            encoded = model.encode_modalities(batch)
            modality_names = ["text", "audio", "visual"]

            real_feat = encoded[modality_to_remove]
            if real_feat is None:
                continue

            # Build condition from other modalities
            cond = model.build_condition(encoded)

            # Diffusion recovery
            recovered = model.diffusion.sample_ddim(
                cond, batch_size=real_feat.size(0), device=device
            )

            # Zero padding
            zero_pad = torch.zeros_like(real_feat)

            # Mean filling (use batch mean of real features)
            mean_fill = real_feat.mean(dim=0, keepdim=True).expand_as(real_feat)

            # Classification with real features
            modality_features_real = []
            conf_list_real = []
            for i, mod in enumerate(modality_names):
                if i == remove_idx:
                    modality_features_real.append(real_feat)
                    conf_list_real.append(1.0)
                elif encoded[mod] is not None:
                    modality_features_real.append(encoded[mod])
                    conf_list_real.append(1.0)
                else:
                    modality_features_real.append(torch.zeros(real_feat.size(0), model.config.model.hidden_dim, device=device))
                    conf_list_real.append(0.0)

            fused_real = model.fusion(modality_features_real, conf_list_real)
            logits_real = model.classifier(fused_real)

            # Classification with diffusion recovered
            modality_features_diff = []
            conf_list_diff = []
            for i, mod in enumerate(modality_names):
                if i == remove_idx:
                    modality_features_diff.append(recovered)
                    conf = model.confidence_eval(recovered, cond)
                    conf_list_diff.append(conf.mean().item())
                elif encoded[mod] is not None:
                    modality_features_diff.append(encoded[mod])
                    conf_list_diff.append(1.0)
                else:
                    modality_features_diff.append(torch.zeros(real_feat.size(0), model.config.model.hidden_dim, device=device))
                    conf_list_diff.append(0.0)

            fused_diff = model.fusion(modality_features_diff, conf_list_diff)
            logits_diff = model.classifier(fused_diff)

            real_feats.append(real_feat)
            zero_pad_feats.append(zero_pad)
            mean_fill_feats.append(mean_fill)
            diffusion_feats.append(recovered)
            real_logits_list.append(logits_real)
            diffusion_logits_list.append(logits_diff)

    if not real_feats:
        return results

    real_cat = torch.cat(real_feats, dim=0)
    zero_cat = torch.cat(zero_pad_feats, dim=0)
    mean_cat = torch.cat(mean_fill_feats, dim=0)
    diff_cat = torch.cat(diffusion_feats, dim=0)
    real_logits_cat = torch.cat(real_logits_list, dim=0)
    diff_logits_cat = torch.cat(diffusion_logits_list, dim=0)

    # Compute metrics
    results["zero_padding"]["mmd"] = compute_mmd(real_cat, zero_cat)
    results["zero_padding"]["cosine_sim"] = compute_cosine_similarity(real_cat, zero_cat)

    results["mean_filling"]["mmd"] = compute_mmd(real_cat, mean_cat)
    results["mean_filling"]["cosine_sim"] = compute_cosine_similarity(real_cat, mean_cat)

    results["diffusion"]["mmd"] = compute_mmd(real_cat, diff_cat)
    results["diffusion"]["cosine_sim"] = compute_cosine_similarity(real_cat, diff_cat)
    results["diffusion"]["cls_consistency"] = compute_classification_consistency(
        real_logits_cat, diff_logits_cat
    )

    return results


def evaluate_modality_contribution(
    model,
    dataloader,
    device: torch.device,
    num_classes: int = 7,
) -> Dict:
    """
    Analyze gating weights to understand modality contributions.
    """
    model.eval()
    gate_weights = {"text": [], "audio": [], "visual": []}
    modality_names = ["text", "audio", "visual"]

    with torch.no_grad():
        for batch in dataloader:
            encoded = model.encode_modalities(batch)
            mask = batch["modality_mask"]

            # Get all modality features
            features = []
            conf_list = []
            for i, mod in enumerate(modality_names):
                if encoded[mod] is not None:
                    features.append(encoded[mod])
                    conf_list.append(1.0)
                else:
                    # Recover
                    cond = model.build_condition(encoded)
                    recovered = model.diffusion.sample_ddim(
                        cond, batch_size=mask.size(0), device=device
                    )
                    features.append(recovered)
                    conf = model.confidence_eval(recovered, cond)
                    conf_list.append(conf.mean().item())

            # Get gating weights
            stacked = torch.stack(features, dim=1)  # (batch, 3, dim)
            x = model.fusion.norm(stacked)
            attn_out, _ = model.fusion.attn(stacked, stacked, stacked)
            x = model.fusion.norm(stacked + attn_out)
            x = x + model.fusion.ffn(x)
            gates = model.fusion.gate(x)  # (batch, 3, 1)
            gates = torch.softmax(gates, dim=1)

            for i, mod in enumerate(modality_names):
                gate_weights[mod].extend(gates[:, i, 0].cpu().numpy().tolist())

    # Compute statistics
    result = {}
    for mod in modality_names:
        vals = np.array(gate_weights[mod])
        result[mod] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    return result
