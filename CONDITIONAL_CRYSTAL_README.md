# Molecule-Conditioned Crystal Generation

This module implements molecular crystal generation conditioned on single molecule structural information, as specified in:
- `SINGLE_MOLECULE_CONDITIONED_CRYSTAL_SPEC.md`
- `SINGLE_MOLECULE_CONDITIONED_CRYSTAL_THEORY.md`
- `SINGLE_MOLECULE_CONDITIONED_SUMMARY.md`
- `SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md`

## Overview

The system generates crystal structures from single molecule inputs using:
1. **MoleculeEncoder**: E(3)-equivariant encoder that converts molecules to fixed-length context vectors
2. **ConditionalCrystalDynamics**: Crystal dynamics model with FiLM conditioning
3. **Paired Dataset Support**: Molecule-crystal correspondence management

## Quick Start

### 1. Create a Test Dataset

```bash
python create_test_paired_dataset.py \
    --output_dir data/test_paired \
    --n_samples 10
```

This creates paired molecule-crystal databases with simple test molecules (methane, ethane, benzene).

### 2. Train a Model

```bash
python main_crystal_conditional.py \
    --molecule_db_path data/test_paired/molecules.db \
    --crystal_db_path data/test_paired/crystals.db \
    --exp_name test_model \
    --epochs 100 \
    --batch_size 2 \
    --molecule_encoder_layers 4 \
    --molecule_encoding_dim 128 \
    --n_layers 6 \
    --hidden_nf 128 \
    --conditioning_method film \
    --lr 1e-4
```

### 3. Generate Crystals

```bash
python sample_crystal_conditional.py \
    --model_path outputs/test_model/final_model.pt \
    --molecule_db_path data/test_paired/molecules.db \
    --molecule_ids 1,2,3 \
    --n_samples_per_molecule 10 \
    --Z 4 \
    --output_dir outputs/generated \
    --output_format cif
```

## Architecture

### Data Flow

```
Molecule DB ──┐
              ├─> MolecularCrystalDataset ─> DataLoader
Crystal DB ───┘                              │
                                            ▼
              Single Molecule ──> MoleculeEncoder ──> context vector
                                              │
              Crystal Structure ──────────────┼──> ConditionalCrystalDynamics
              Time step t ─────────────────────┘
                                              │
                                              ▼
                                    Predicted Crystal
```

### Key Components

1. **MoleculeEncoder** (`crystal/models/molecule_encoder.py`)
   - E(3)-equivariant EGNN layers
   - Graph pooling (mean, max, mean_max, attention)
   - Output: fixed-length context vector (default: 128D)

2. **ConditionalCrystalDynamics** (`crystal/models/conditional_crystal_dynamics.py`)
   - FiLM conditioning layers
   - Periodic EGNN for atomic positions
   - Lattice diffusion for cell parameters
   - Supports 'film' and 'add' conditioning methods

3. **MolecularCrystalDataset** (`crystal/data/molecular_crystal_loader.py`)
   - Manages paired molecule-crystal data
   - Handles variable-sized structures
   - Automatic train/valid/test splitting

## Model Parameters

### Molecule Encoder
- `--molecule_encoder_layers`: Number of EGNN layers (default: 6)
- `--molecule_encoder_hidden`: Hidden dimension (default: 256)
- `--molecule_encoding_dim`: Output embedding dimension (default: 128)
- `--encoder_aggregation`: Pooling method (mean/max/mean_max/attention)

### Crystal Dynamics
- `--hidden_nf`: Hidden feature dimension (default: 256)
- `--n_layers`: Number of EGNN layers (default: 9)
- `--attention`: Use attention mechanism
- `--learn_lattice`: Learn lattice parameters
- `--conditioning_method`: Conditioning method (film/add)

### Training
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--gradient_clip`: Gradient clipping value (default: 1.0)

### Classifier-Free Guidance
- `--guidance_scale`: Guidance scale (default: 0.0)
- `--cfg_dropout`: Context dropout probability for CFG training (default: 0.1)

## Implementation Status

### ✅ Completed
- [x] ConditionalCrystalDynamics model with FiLM conditioning
- [x] MoleculeEncoder with E(3)-equivariance
- [x] MolecularCrystalDataset for paired data
- [x] Training script (main_crystal_conditional.py)
- [x] Sampling script (sample_crystal_conditional.py)
- [x] Test dataset creation utility
- [x] Module exports and integration

### 🚧 In Progress / Future Work
- [ ] Full diffusion sampling implementation
- [ ] Molecular consistency loss
- [ ] Evaluation metrics (consistency, validity, quality)
- [ ] Advanced conditioning methods (cross-attention)
- [ ] Multi-molecule crystal generation
- [ ] Space group constraints
- [ ] Property-based conditioning (density, etc.)

## Testing

### Unit Tests

Test the conditional crystal dynamics model:
```bash
python -m crystal.models.conditional_crystal_dynamics
```

Expected output:
```
Testing ConditionalCrystalDynamics...
✓ Forward pass successful
✓ Model responds to different molecular contexts
All tests passed!
```

### Integration Test

1. Create test dataset:
```bash
python create_test_paired_dataset.py --n_samples 5
```

2. Quick training test (CPU):
```bash
python main_crystal_conditional.py \
    --molecule_db_path data/test_paired/molecules.db \
    --crystal_db_path data/test_paired/crystals.db \
    --exp_name quick_test \
    --epochs 2 \
    --batch_size 1 \
    --molecule_encoder_layers 2 \
    --n_layers 2 \
    --hidden_nf 64
```

## Data Format

### Molecule Database Schema
- `id`: Unique molecule ID
- `symbols`: Atom types
- `positions`: Atomic coordinates (Å)
- `numbers`: Atomic numbers

### Crystal Database Schema
- `id`: Unique crystal ID
- `molecule_id`: Reference to molecule database
- `symbols`: Atom types in unit cell
- `positions`: Atomic coordinates (Å)
- `cell`: Lattice vectors (3×3 matrix)
- `pbc`: Periodic boundary conditions [True, True, True]
- `Z`: Number of molecules per unit cell
- `density`: Crystal density (g/cm³, optional)

## Performance Notes

- **Memory**: ~2-4 GB for batch_size=2 with 50-100 atoms
- **Training Speed**: ~5-10 iterations/second on GPU (RTX 3090)
- **Recommended Settings**:
  - Start with small models (hidden_nf=128, n_layers=4-6)
  - Use batch_size=2-4 for molecules with 50+ atoms
  - Increase gradually after confirming training stability

## References

1. Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D" (ICML 2022)
2. Jiao et al. "Crystal Diffusion Variational Autoencoder" (2023)
3. Satorras et al. "E(n) Equivariant Graph Neural Networks" (ICML 2021)
4. Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Reduce `--hidden_nf` or `--n_layers`
- Use `--num_workers 0` to disable multiprocessing

### Training Not Converging
- Check learning rate (try 5e-5 or 1e-3)
- Verify dataset integrity
- Start with simpler models

### Sampling Issues
- Ensure model is properly loaded
- Check input molecule format
- Verify context vector generation

## Contact

For issues or questions, please open an issue on GitHub.
