# FedDiff-MSA

**Federated Diffusion-based Multimodal Sentiment Analysis**

A novel framework that integrates conditional diffusion models, risk-aware layered differential privacy, and confidence-aware fusion for federated multimodal sentiment analysis under modality-missing scenarios.

## Overview

FedDiff-MSA addresses three key challenges in federated multimodal learning:

1. **Modality heterogeneity** — Clients may lack certain modalities due to device/sensor limitations
2. **Privacy preservation** — Gradient updates leak sensitive information, especially for high-memorization components like diffusion models
3. **Communication efficiency** — Federated rounds are expensive; model updates must be compressed

### Key Contributions

- **Feature-space conditional diffusion** for missing modality recovery (256-dim, end-to-end emotion-aligned)
- **Triple loss**: denoising MSE + emotion consistency (L_emo) + cross-modal contrastive (L_contrast)
- **Truncated backpropagation through time (TBTT)** for efficient L_emo gradient flow (T_bp=10)
- **MAFA-Diff aggregation**: quality-aware weighted averaging with momentum compensation
- **Risk-aware layered differential privacy**: differentiated DP budgets per parameter group
- **Confidence-aware fusion**: gating weights conditioned on recovery quality
- **Communication optimization**: Top-K sparsification + INT8 quantization with error feedback

## Project Structure

```
feddiff_msa/
├── config/
│   └── config.py                 # Centralized configuration (ModelConfig, FederatedConfig, etc.)
├── data/
│   └── data_loader.py            # Dataset loading, Dirichlet partitioning, modality-missing simulation
├── models/
│   ├── diffusion.py              # Conditional U-Net + cosine schedule + DDIM sampling + TBTT
│   ├── model.py                  # FedDiffModel: encoders + diffusion + fusion + classifier + confidence
│   └── losses.py                 # TripleLoss: denoising + emotion consistency + contrastive
├── federated/
│   └── aggregation.py            # MAFA-Diff aggregation + layered DP + communication compression + RDP
├── training/
│   └── trainer.py                # FederatedClient + FederatedTrainer: training loop
├── utils/
│   └── metrics.py                # Classification metrics, MMD, cosine sim, recovery quality, modality contribution
├── run_experiments.py            # Experiment runner (6 experiments → 7 paper tables)
├── generate_figures.py           # Figure generation (6 data-driven figures)
└── README.md
```

## Experiments

### Table-Output Experiments (`run_experiments.py`)

| Experiment | Command | Paper Table | Description |
|------------|---------|-------------|-------------|
| Main results | `--exp main` | Table 3 | 9 methods × 3 settings × {Acc-2, F1-7} |
| Recovery quality | `--exp recovery` | Table 4 | 4 recovery methods × 3 metrics |
| Ablation study | `--exp ablation` | Table 5 | 11 variants × {Acc-2, F1-7} |
| Privacy-utility | `--exp privacy` | Figure 6 data | Layered DP vs. Uniform DP across ε |
| MIA success rates | `--exp mia` | Table 6 | 5 parameter groups × 2 DP modes |
| Client scalability | `--exp scalability` | Table 7 | K ∈ {5,10,20,50,100} × 3 metrics |

### Figure Generation (`generate_figures.py`)

| Figure | Command | Description |
|--------|---------|-------------|
| Fig 2 | `--fig convergence` | Training convergence curves (5 methods × 3 settings) |
| Fig 3 | `--fig tsne` | t-SNE visualization of recovered features (4 subplots) |
| Fig 4 | `--fig noniid` | Performance vs. Dirichlet α (4 methods) |
| Fig 5 | `--fig missing_ratio` | Performance vs. modality missing ratio (3 methods) |
| Fig 6 | `--fig privacy_utility` | Privacy-utility Pareto curve (Layered vs. Uniform DP) |
| Fig 7 | `--fig contribution` | Modality contribution heatmap (gating weights) |

## Quick Start

### Local CPU Test (No GPU Required)

```bash
# Run all experiments with tiny synthetic data
python run_experiments.py --test

# Run individual experiments
python run_experiments.py --test --exp main
python run_experiments.py --test --exp ablation

# Generate all figures
python generate_figures.py --test

# Generate individual figure
python generate_figures.py --test --fig convergence
```

### Full Experiment on GPU Server

```bash
# 1. Prepare pre-extracted features
#    Place in data/features/:
#      mosei_text.npy    (N, 768)   — RoBERTa-base
#      mosei_audio.npy   (N, 768)   — Wav2Vec2-base
#      mosei_visual.npy  (N, 342)   — OpenFace 2.0
#      mosei_labels.npy  (N,)

# 2. Run experiments
python run_experiments.py --dataset mosei --setting C --device cuda

# 3. Generate figures
python generate_figures.py --device cuda
```

### Output Files

Each experiment produces:

- **CSV table files** (`table_XX_*.csv`) — Directly mapped to paper tables, with metadata comments
- **JSON detail files** (`results_*.json`) — Raw results for further analysis
- **PDF + PNG figures** (`figures/fig*_*.pdf/.png`) — Vector graphics for paper submission
- **README_results.md** — Index of all output files and their paper correspondence

## Configuration

All hyperparameters are defined in `config/config.py`:

| Component | Key Parameters |
|-----------|---------------|
| Model | hidden_dim=256, diffusion_steps=50, num_classes=7, tbtt_window=10 |
| Federated | K=20 clients, 100 rounds, 5 local epochs, Dirichlet α=0.5 |
| Privacy | ε=8, δ=1e-5, layered budgets (40% diffusion, 15% each others) |
| Communication | Top-K (10%), INT8 quantization, error feedback |

## Requirements

```
torch>=2.0
numpy
scikit-learn
matplotlib
```

## Target Venues

- Information Fusion (IF ~18)
- IEEE Transactions on Affective Computing (IF ~13)

## License

Research code. Not for redistribution without permission.
