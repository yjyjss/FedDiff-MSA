"""
FedDiff-MSA Configuration
Centralized configuration for all experiments.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class FederatedSetting(str, Enum):
    IID = "A"
    NON_IID = "B"
    MODALITY_MISSING = "C"


class DatasetName(str, Enum):
    CMU_MOSEI = "mosei"
    IEMOCAP = "iemocap"
    MELD = "meld"


@dataclass
class ModelConfig:
    # Feature extractors (frozen)
    text_dim: int = 768        # RoBERTa-base output
    audio_dim: int = 768       # Wav2Vec2-base output
    visual_dim: int = 342      # OpenFace 2.0 output

    # Modality encoders
    hidden_dim: int = 256      # Unified representation dimension

    # Diffusion model
    diffusion_steps: int = 50
    diffusion_cond_dim: int = 256
    unet_layers: int = 3       # ResBlocks
    unet_heads: int = 2        # Cross-attention heads
    time_embed_dim: int = 128
    diffusion_eta: int = 0     # DDIM deterministic

    # Emotion classifier
    num_classes: int = 7       # MOSEI 7-class
    num_heads: int = 8         # Cross-modal attention heads
    classifier_hidden: int = 128
    dropout: float = 0.1

    # Confidence evaluator
    conf_hidden: int = 128

    # Loss weights
    lambda_diff: float = 1.0
    lambda_emo: float = 0.5
    lambda_contrast: float = 0.1
    temperature: float = 0.07

    # MAFA-Diff
    gamma: float = 0.2         # Quality bonus
    beta: float = 0.9          # Momentum coefficient

    # Truncation for L_emo backprop
    tbtt_window: int = 10


@dataclass
class FederatedConfig:
    num_clients: int = 20
    num_rounds: int = 100
    local_epochs: int = 5
    sampling_rate: float = 0.3
    early_stop_patience: int = 10
    dirichlet_alpha: float = 0.5

    # Modality missing (Setting C)
    missing_visual_rate: float = 0.3
    missing_audio_rate: float = 0.2
    text_only_rate: float = 0.1

    # FedProx proximal term (0 = standard FedAvg)
    prox_mu: float = 0.0

    # Aggregation strategy
    aggregation: str = "mafa_diff"  # "mafa_diff", "fedavg", "fedmm_sa"


@dataclass
class PrivacyConfig:
    enabled: bool = True
    target_epsilon: float = 8.0
    delta: float = 1e-5
    client_sample_rate: float = 0.3

    # Layered DP budgets (must sum to 1.0)
    budget_text: float = 0.15
    budget_audio: float = 0.15
    budget_visual: float = 0.15
    budget_diffusion: float = 0.40
    budget_fusion: float = 0.15

    # Clip norms per group
    clip_text: float = 1.0
    clip_audio: float = 1.0
    clip_visual: float = 1.0
    clip_diffusion: float = 0.5
    clip_fusion: float = 1.5

    # Noise multipliers per group
    sigma_text: float = 0.3
    sigma_audio: float = 0.3
    sigma_visual: float = 0.3
    sigma_diffusion: float = 0.8
    sigma_fusion: float = 0.2


@dataclass
class CommunicationConfig:
    enabled: bool = True
    sparsification_ratio: float = 0.1  # Top-K
    quantization_bits: int = 8         # INT8
    error_feedback: bool = True


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    num_workers: int = 2
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level config combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    dataset: str = "mosei"
    setting: str = "C"
    exp_name: str = "feddiff_msa"
    output_dir: str = "./outputs"
    device: str = "auto"  # "auto", "cpu", "cuda"
    test_mode: bool = False  # If True, use tiny synthetic data for local testing

    # Data paths
    data_root: str = "./data"
    feature_root: str = "./data/features"

    # Missing modality handling strategy for baselines
    missing_strategy: str = "zero_pad"  # "zero_pad", "mean_fill", "autoencoder"

    # Synthetic data size (test mode only)
    num_samples_per_client: int = 200


def get_test_config() -> ExperimentConfig:
    """Return a lightweight config for local CPU testing."""
    cfg = ExperimentConfig()
    cfg.test_mode = True
    cfg.device = "cpu"
    cfg.model.num_classes = 3
    cfg.model.diffusion_steps = 5       # Reduced for speed
    cfg.model.tbtt_window = 2
    cfg.model.hidden_dim = 32           # Tiny model
    cfg.model.text_dim = 32
    cfg.model.audio_dim = 32
    cfg.model.visual_dim = 32
    cfg.model.time_embed_dim = 16
    cfg.model.classifier_hidden = 32
    cfg.federated.num_clients = 3
    cfg.federated.num_rounds = 2
    cfg.federated.local_epochs = 1
    cfg.federated.sampling_rate = 1.0   # All clients participate
    cfg.federated.dirichlet_alpha = 0.5
    cfg.training.batch_size = 8
    cfg.training.num_workers = 0
    cfg.privacy.enabled = True
    cfg.privacy.target_epsilon = 8.0
    cfg.communication.enabled = False    # Skip for test speed
    cfg.output_dir = "./outputs_test"
    return cfg


def get_default_config() -> ExperimentConfig:
    """Return the full experiment config matching the technical proposal."""
    return ExperimentConfig()
