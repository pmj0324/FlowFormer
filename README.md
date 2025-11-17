# FlowFormer - IceCube Neutrino Event Generation

Continuous Normalizing Flow (CNF) models for IceCube neutrino event generation.

## Models

Two model architectures are available:

1. **RelativeGeometryCNF** (Option B): Transformer-based model with geometry-aware attention
2. **EquivariantCNF** (Option C): SE(3)-equivariant model (EGNN-style)

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Format

The HDF5 file should contain:
- `input`: (N_events, 2, N_DOMs) - DOM signals [time, charge]
- `label`: (N_events, 6) - Event parameters [E, θ, φ, x, y, z]
- `xpmt`, `ypmt`, `zpmt`: (N_DOMs,) - DOM positions

## Usage

### 1. Configure Training

Edit `config.yaml` to set:
- Data paths
- Model type and hyperparameters
- Training settings
- Device (cuda/cpu)

### 2. Train Model

```bash
python train.py -c config.yaml
```

Or use default config:
```bash
python train.py
```

### 3. Monitor Training

Training progress will be shown in the terminal. If WandB is enabled in config, metrics will be logged online.

Checkpoints are saved to `./checkpoints/` by default.

## Configuration Options

### Model Types

- `RelativeGeometryCNF`: Transformer with geometric bias
  - Better for capturing long-range dependencies
  - Higher memory usage
  
- `EquivariantCNF`: SE(3)-equivariant network
  - Respects rotational symmetries
  - More parameter efficient

### Training Tips

- Start with smaller batch sizes (16-32) to avoid OOM
- Use gradient clipping (default: 1.0) for stability
- Monitor validation loss for overfitting
- Adjust learning rate based on loss curve

## Files

- `model.py`: Model architectures
- `dataset.py`: Data loading and preprocessing
- `train.py`: Training script
- `config.yaml`: Configuration file
- `h5_reader.py`: Utility to inspect HDF5 files

## Example

```bash
# Inspect data
python h5_reader.py -p ./data/22644.h5

# Train with default config
python train.py

# Train with custom config
python train.py -c my_config.yaml
```

## Output

Training produces:
- `checkpoints/`: Model checkpoints
  - `checkpoint_epoch_N.pth`: Regular checkpoints
  - `checkpoint_epoch_N_best.pth`: Best model (lowest val loss)
- `checkpoints/normalization_stats.pth`: Data normalization parameters
- `checkpoints/config.yaml`: Copy of training config

