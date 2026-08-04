"""
FedDiff-MSA Data Module
Handles dataset loading, federated partitioning, and synthetic data for testing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config.config import ExperimentConfig, FederatedSetting


@dataclass
class ModalitySample:
    text: Optional[torch.Tensor]      # (text_dim,) or None
    audio: Optional[torch.Tensor]     # (audio_dim,) or None
    visual: Optional[torch.Tensor]    # (visual_dim,) or None
    label: int
    modality_mask: np.ndarray         # (3,) binary: [text, audio, visual]


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal sentiment analysis.
    Supports loading pre-extracted features or generating synthetic data.
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
        modality_masks: Optional[np.ndarray] = None,
    ):
        """
        Args:
            features: dict with keys 'text', 'audio', 'visual', each (N, dim) or None
            labels: (N,) integer labels
            modality_masks: (N, 3) binary masks, None means all modalities available
        """
        self.features = features
        self.labels = labels
        self.n_samples = len(labels)

        if modality_masks is None:
            modality_masks = np.ones((self.n_samples, 3), dtype=np.float32)

        self.modality_masks = modality_masks

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> Dict:
        mask = self.modality_masks[idx]

        text = None
        audio = None
        visual = None

        if mask[0] > 0 and "text" in self.features and self.features["text"] is not None:
            text = torch.FloatTensor(self.features["text"][idx])

        if mask[1] > 0 and "audio" in self.features and self.features["audio"] is not None:
            audio = torch.FloatTensor(self.features["audio"][idx])

        if mask[2] > 0 and "visual" in self.features and self.features["visual"] is not None:
            visual = torch.FloatTensor(self.features["visual"][idx])

        return {
            "text": text,
            "audio": audio,
            "visual": visual,
            "label": torch.LongTensor([self.labels[idx]]).squeeze(),
            "modality_mask": torch.FloatTensor(mask),
            "idx": idx,
        }


def generate_synthetic_data(
    n_samples: int = 200,
    text_dim: int = 32,
    audio_dim: int = 32,
    visual_dim: int = 32,
    num_classes: int = 3,
    missing_rate: float = 0.3,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Generate synthetic multimodal data for testing.
    Features are class-conditional to allow a model to learn above chance.
    """
    rng = np.random.RandomState(seed)

    labels = rng.randint(0, num_classes, size=n_samples)

    # Class-conditional features with clear separation
    centers = rng.randn(num_classes, max(text_dim, audio_dim, visual_dim))

    text = centers[labels, :text_dim] + 0.5 * rng.randn(n_samples, text_dim)
    audio = centers[labels, :audio_dim] + 0.5 * rng.randn(n_samples, audio_dim)
    visual = centers[labels, :visual_dim] + 0.5 * rng.randn(n_samples, visual_dim)

    features = {
        "text": text.astype(np.float32),
        "audio": audio.astype(np.float32),
        "visual": visual.astype(np.float32),
    }

    # Simulate modality missing
    masks = np.ones((n_samples, 3), dtype=np.float32)
    for i in range(n_samples):
        r = rng.random()
        if r < missing_rate:
            missing_modality = rng.randint(0, 3)
            masks[i, missing_modality] = 0.0

    return features, labels, masks


def partition_data_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Partition data indices across clients using Dirichlet distribution.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(labels)
    num_classes = len(np.unique(labels))

    # Generate proportions per client per class
    client_indices = []

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet([alpha] * num_clients)
        # Split class indices according to proportions
        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, splits)

        if len(client_indices) == 0:
            client_indices = [[] for _ in range(num_clients)]

        for i, split in enumerate(class_splits):
            client_indices[i].extend(split.tolist())

    # Shuffle each client's indices
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i])

    return client_indices


def partition_data_iid(
    labels: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """Uniform random partition."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(labels))
    return [indices[i::num_clients] for i in range(num_clients)]


def apply_modality_missing(
    modality_masks: np.ndarray,
    config: ExperimentConfig,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply Setting C: simulate modality absence at the client level.
    Some clients lose visual, some lose audio, some are text-only.
    """
    rng = np.random.RandomState(seed)
    masks = modality_masks.copy()

    n = len(masks)
    n_missing_visual = int(n * config.federated.missing_visual_rate)
    n_missing_audio = int(n * config.federated.missing_audio_rate)
    n_text_only = int(n * config.federated.text_only_rate)

    # Randomly assign missing patterns
    perm = rng.permutation(n)
    idx = 0

    # Missing visual
    for i in range(idx, idx + n_missing_visual):
        masks[perm[i], 2] = 0.0
    idx += n_missing_visual

    # Missing audio
    for i in range(idx, idx + n_missing_audio):
        masks[perm[i], 1] = 0.0
    idx += n_missing_audio

    # Text-only
    for i in range(idx, idx + n_text_only):
        masks[perm[i], 1] = 0.0
        masks[perm[i], 2] = 0.0

    return masks


def create_federated_clients(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    modality_masks: np.ndarray,
    config: ExperimentConfig,
) -> List[MultimodalDataset]:
    """
    Create per-client datasets based on federated partitioning strategy.
    """
    num_clients = config.federated.num_clients
    alpha = config.federated.dirichlet_alpha
    setting = config.setting

    if setting == "A":  # IID
        client_indices = partition_data_iid(labels, num_clients, seed=config.training.seed)
    else:  # Non-IID or Modality-Missing
        client_indices = partition_data_dirichlet(
            labels, num_clients, alpha=alpha, seed=config.training.seed
        )

    if setting == "C":  # Apply modality missing
        modality_masks = apply_modality_missing(modality_masks, config, seed=config.training.seed)

    client_datasets = []
    for indices in client_indices:
        if len(indices) == 0:
            # Ensure at least a few samples
            indices = np.random.choice(len(labels), size=min(5, len(labels)), replace=False)

        client_features = {k: v[indices] if v is not None else None for k, v in features.items()}
        client_labels = labels[indices]
        client_masks = modality_masks[indices]
        client_datasets.append(MultimodalDataset(client_features, client_labels, client_masks))

    return client_datasets


def multimodal_collate_fn(batch):
    """Custom collate function that handles None modality values."""
    result = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [item[key] for item in batch]
        if key in ("text", "audio", "visual"):
            # Check if all values are None (modality missing for entire batch)
            if all(v is None for v in vals):
                result[key] = None
            else:
                # Replace None with zero tensors, keep track via modality_mask
                # Find first non-None to get shape
                sample = next(v for v in vals if v is not None)
                tensors = [v if v is not None else torch.zeros_like(sample) for v in vals]
                result[key] = torch.stack(tensors, dim=0)
        elif key == "label":
            result[key] = torch.stack(vals, dim=0)
        elif key == "modality_mask":
            result[key] = torch.stack(vals, dim=0)
        elif key == "idx":
            result[key] = torch.LongTensor(vals)
        else:
            result[key] = vals
    return result


def build_dataloaders(
    client_datasets: List[MultimodalDataset],
    batch_size: int,
    num_workers: int = 0,
) -> List[DataLoader]:
    """Create DataLoader for each client."""
    loaders = []
    for ds in client_datasets:
        loaders.append(DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=False,
            collate_fn=multimodal_collate_fn,
        ))
    return loaders


def prepare_data(config: ExperimentConfig):
    """
    Main entry: prepare federated client datasets and test set.
    Returns (client_dataloaders, test_dataset, test_dataloader).
    """
    if config.test_mode:
        features, labels, masks = generate_synthetic_data(
            n_samples=200,
            text_dim=config.model.text_dim,
            audio_dim=config.model.audio_dim,
            visual_dim=config.model.visual_dim,
            num_classes=config.model.num_classes,
            missing_rate=0.3,
            seed=config.training.seed,
        )
    else:
        # Load real pre-extracted features
        # Expected files in config.feature_root:
        #   mosei_text.npy    (N, 768)
        #   mosei_audio.npy   (N, 768)
        #   mosei_visual.npy  (N, 342)
        #   mosei_labels.npy  (N,)
        import os
        ds_name = config.dataset
        text = np.load(os.path.join(config.feature_root, f"{ds_name}_text.npy"))
        audio = np.load(os.path.join(config.feature_root, f"{ds_name}_audio.npy"))
        visual = np.load(os.path.join(config.feature_root, f"{ds_name}_visual.npy"))
        labels = np.load(os.path.join(config.feature_root, f"{ds_name}_labels.npy"))
        masks = np.ones((len(labels), 3), dtype=np.float32)
        features = {"text": text, "audio": audio, "visual": visual}

    num_clients = config.federated.num_clients

    # Split into train (80%) and test (20%)
    n = len(labels)
    rng = np.random.RandomState(config.training.seed)
    test_size = max(int(n * 0.2), num_clients)
    test_indices = rng.choice(n, size=test_size, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_indices] = False

    train_features = {k: v[train_mask] if v is not None else None for k, v in features.items()}
    train_labels = labels[train_mask]
    train_masks = masks[train_mask]

    test_features = {k: v[test_indices] if v is not None else None for k, v in features.items()}
    test_labels = labels[test_indices]
    test_masks = masks[test_indices]

    # Create federated clients
    client_datasets = create_federated_clients(train_features, train_labels, train_masks, config)
    client_dataloaders = build_dataloaders(client_datasets, config.training.batch_size, config.training.num_workers)

    # Test set
    test_dataset = MultimodalDataset(test_features, test_labels, test_masks)
    test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, collate_fn=multimodal_collate_fn)

    return client_dataloaders, test_dataset, test_dataloader
