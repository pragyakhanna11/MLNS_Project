# Physics-Informed Variational Inference for Cell Migration

This repository contains the implementation of physics-informed variational inference models for analyzing and predicting cell migration trajectories. Three different model architectures are provided:

1. **Base PIVAE**: Physics-Informed Variational Autoencoder with standard encoder-decoder architecture
2. **HNN-PIVAE**: Extension with Hamiltonian Neural Networks for oscillatory dynamics
3. **SDE-PIVAE**: Extension with Neural Stochastic Differential Equations for state-dependent noise
4. **NF-PIVAE**: Extension with Normalizing Flows for improved uncertainty quantification

## Project Structure

The repository is organized as follows:

```
.
├── data/                          # Data loading and preprocessing
│   ├── __init__.py
│   ├── loader.py                  # Functions to load cell migration data
│   ├── preprocessing.py           # Functions for data normalization and cleaning
│   └── visualization.py           # Functions to visualize trajectories
│
├── models/                        # Model implementations
│   ├── base/                      # Basic components for all models
│   │   ├── encoder.py             # Base encoder architecture
│   │   ├── decoder.py             # Base decoder with basis and modal networks
│   │   ├── layers.py              # Common layer definitions
│   │   └── resnet.py              # ResNet blocks
│   ├── composite/                 # Complete model implementations
│   │   ├── pivae.py               # Standard Physics-Informed VAE
│   │   ├── pi_hnn.py              # Physics-Informed VAE with HNN
│   │   ├── pi_sde.py              # Physics-Informed VAE with Neural SDE
│   │   └── pi_nf.py               # Physics-Informed VAE with Normalizing Flows
│   ├── hamiltonian/               # Hamiltonian Neural Network components
│   │   ├── hnn_encoder.py         # HNN-based encoder
│   │   └── utils.py               # HNN utilities
│   ├── neural_sde/                # Neural SDE components
│   │   ├── sde_encoder.py         # SDE-based encoder
│   │   └── diffusion.py           # SDE noise models and integrators
│   └── normalizing_flow/          # Normalizing Flow components
│       ├── flow_decoder.py        # Flow-based decoder
│       └── transforms.py          # Flow transformations (RealNVP, MAF, etc.)
│
├── physics/                       # Physics-based model components
│   ├── hamiltonian.py             # Hamiltonian dynamics residuals
│   ├── neural_sde.py              # Neural SDE residuals
│   ├── normalizing_flow.py        # Normalizing Flow residuals
│   └── ou_process.py              # Ornstein-Uhlenbeck process
│
├── scripts/                       # Training scripts
│   ├── prepare_data.py            # Data preparation script
│   ├── train_base.py              # Script for training base model
│   ├── train_hnn.py               # Script for training HNN model
│   ├── train_sde.py               # Script for training Neural SDE model
│   └── train_nf.py                # Script for training Normalizing Flow model
│
└── training/                      # Training utilities
    ├── loss.py                    # Loss functions (ELBO, physics residuals)
    └── trainer.py                 # Trainer class for model training
```

## Installation

### Requirements

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/pivae-cell-migration.git
cd pivae-cell-migration
pip install -r requirements.txt
```

## Data Preparation

The main dataset used in this project is the confine cell migration dataset, which contains tracked trajectories of 149 cells over time.

To prepare the data for training:

```bash
python scripts/prepare_data.py --input path/to/trajectories_x.txt --output prepared_data/processed_trajectories.npy
```

This script will:
1. Load the raw trajectory data
2. Handle missing values through interpolation
3. Normalize the trajectories
4. Save the processed data for training

## Model Training

### Base PIVAE Model

The base Physics-Informed VAE model uses a standard encoder and a structured decoder with basis and modal networks.

```bash
python scripts/train_base.py \
    --input prepared_data/processed_trajectories.npy \
    --output-dir results/base_model \
    --window-size 20 \
    --latent-dim 16 \
    --hidden-dim 128 \
    --output-dim 64 \
    --batch-size 32 \
    --epochs 500 \
    --lr 1e-4 \
    --physics-model persistence
```

### Hamiltonian Neural Network (HNN) Model

The HNN model extends the base model by using a Hamiltonian Neural Network as the encoder, which is particularly suitable for modeling oscillatory behaviors like those observed in cancer cell migration.

```bash
python scripts/train_hnn.py \
    --input prepared_data/processed_trajectories.npy \
    --output-dir results/hnn_model \
    --window-size 20 \
    --latent-dim 16 \
    --hidden-dim 128 \
    --output-dim 64 \
    --batch-size 32 \
    --epochs 500 \
    --lr 1e-4 \
    --physics-model hamiltonian
```

### Neural SDE Model

The Neural SDE model uses a stochastic differential equation in the latent space to model state-dependent noise, providing more accurate uncertainty quantification.

```bash
python scripts/train_sde.py \
    --input prepared_data/processed_trajectories.npy \
    --output-dir results/sde_model \
    --window-size 20 \
    --latent-dim 8 \
    --hidden-dim 64 \
    --output-dim 32 \
    --batch-size 32 \
    --epochs 500 \
    --lr 1e-4 \
    --physics-model neural_sde
```

### Normalizing Flow Model

The Normalizing Flow model uses invertible transformations to transform a simple prior distribution into a more complex one, allowing for better modeling of complex uncertainty patterns.

```bash
python scripts/train_nf.py \
    --input prepared_data/processed_trajectories.npy \
    --output-dir results/nf_model \
    --window-size 20 \
    --latent-dim 8 \
    --hidden-dim 64 \
    --output-dim 32 \
    --flow-type realnvp \
    --num-flows 3 \
    --batch-size 32 \
    --epochs 500 \
    --lr 1e-4 \
    --physics-model normalizing_flow
```

## Model Parameters

Each model can be customized with the following parameters:

- `--window-size`: Size of the sliding window for trajectory segments (default: 20)
- `--latent-dim`: Dimension of the latent space (default: 8 or 16)
- `--hidden-dim`: Dimension of hidden layers in networks (default: 64 or 128)
- `--output-dim`: Output dimension of basis and modal networks (default: 32 or 64)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 500)
- `--lr`: Learning rate (default: 1e-4)
- `--use-physics`: Whether to use physics constraints (default: True)
- `--progressive-physics`: Gradually increase physics weight during training (default: True)
- `--physics-model`: Physics model to use (options: "ou", "persistence", "hamiltonian", "neural_sde", "normalizing_flow")
- `--gpu`: Use GPU for training if available

For the Normalizing Flow model, additional parameters include:
- `--flow-type`: Type of normalizing flow to use (options: "planar", "realnvp", "maf")
- `--num-flows`: Number of flow layers (default: 3)

## Model Evaluation

After training, you can evaluate and visualize the results using:

```bash
# For the base model
python scripts/evaluate_model.py --model-type base --model-path results/base_model/base_best_model.pt --input prepared_data/test_trajectories.npy --output-dir results/base_evaluation

# For the HNN model
python scripts/evaluate_model.py --model-type hnn --model-path results/hnn_model/hnn_best_model.pt --input prepared_data/test_trajectories.npy --output-dir results/hnn_evaluation

# For the SDE model
python scripts/evaluate_model.py --model-type sde --model-path results/sde_model/sde_best_model.pt --input prepared_data/test_trajectories.npy --output-dir results/sde_evaluation

# For the NF model
python scripts/evaluate_model.py --model-type nf --model-path results/nf_model/nf_best_model.pt --input prepared_data/test_trajectories.npy --output-dir results/nf_evaluation
```

## Results Visualization

Each training script automatically generates visualizations of:

1. Training loss curves
2. Reconstruction quality
3. Uncertainty quantification
4. Latent space structure
5. Model-specific visualizations:
   - **HNN**: Hamiltonian energy and phase space trajectories
   - **SDE**: Stochastic paths and state-dependent noise
   - **NF**: Flow transformations and probability density evolution

These visualizations are saved in the specified output directory and can be used to analyze the model's performance and behavior.

