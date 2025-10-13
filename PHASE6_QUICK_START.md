# Phase 6: Crystal Training Loop - Quick Start Guide

## Overview

This guide shows how to use the newly implemented crystal-specific training loop (Phase 6).

## Installation

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pytest wandb ase scipy tqdm imageio
```

## Running Tests

```bash
# Run Phase 6 tests
cd /path/to/e3_diffusion_for_molecules
PYTHONPATH=.:$PYTHONPATH python tests/test_training_loop.py

# Expected: All tests pass with some NotImplementedError notes (expected)
```

## Training a Crystal Model

### Minimum Configuration

```bash
python main_crystal.py \
    --exp_name my_first_crystal \
    --crystal_db_path data/crystals.db \
    --molecule_db_path data/molecules.db \
    --condition_on_molecule True \
    --batch_size 16 \
    --n_epochs 10 \
    --test_epochs 2 \
    --no_wandb
```

### Full Configuration Example

```bash
python main_crystal.py \
    --exp_name full_featured_crystal \
    --crystal_db_path data/crystals.db \
    --molecule_db_path data/molecules.db \
    \
    # Conditioning
    --condition_on_molecule True \
    --condition_on_space_group False \
    --condition_on_density False \
    --conditioning_dim 256 \
    \
    # Model architecture
    --n_layers 6 \
    --nf 128 \
    --attention True \
    --learn_lattice True \
    --lattice_hidden_dim 128 \
    \
    # Training
    --batch_size 32 \
    --n_epochs 200 \
    --lr 1e-4 \
    --test_epochs 10 \
    --ema_decay 0.999 \
    \
    # Evaluation
    --n_stability_samples 100 \
    --validate_structures True \
    --save_cif True \
    \
    # Logging
    --no_wandb \
    --n_report_steps 10
```

## Key Components

### 1. Training Function

```python
from train_test_crystal import train_epoch_crystal

# Called automatically by main_crystal.py
# Handles:
# - Periodic boundary conditions
# - Cell parameter learning
# - Molecular conditioning
# - Gradient clipping
# - EMA updates
```

### 2. Validation Function

```python
from train_test_crystal import test_crystal

# Called automatically during validation
# Computes NLL on validation set
```

### 3. Analysis Function

```python
from train_test_crystal import analyze_and_save_crystal

# Generates crystals and computes metrics
# Note: Sampling not yet fully implemented
```

### 4. Sampling Functions

```python
from crystal.sampling import (
    sample_crystal,
    sample_crystal_chain,
    validate_and_save_crystal
)

# Status: Interface defined, implementation pending
# Will raise NotImplementedError until diffusion sampling is integrated
```

## What Works Now

✅ **Training Loop**
- Complete epoch training
- Loss computation (using molecular version)
- Gradient clipping
- EMA model updates
- Checkpoint saving

✅ **Validation**
- Validation epoch
- NLL computation
- Metric logging

✅ **Configuration**
- Molecular conditioning
- Space group conditioning (requires molecular)
- Density conditioning (requires molecular)
- All command-line arguments

✅ **Error Handling**
- Proper ValueError for missing configurations
- NotImplementedError for unfinished features
- No silent fallbacks

## What's Pending (Phase 7)

⏳ **Crystal Sampling**
- Reverse-time diffusion integration
- Cell parameter sampling
- Joint position/cell generation

⏳ **Crystal Loss Computation**
- Cell parameter loss term
- Crystal-specific regularization
- Proper gradient flow for cell

⏳ **Node Distribution**
- Learn from crystal size distribution
- Currently placeholder (None)

## Troubleshooting

### Error: "mol_encoder is None"

**Cause**: Trying to use molecular conditioning without providing molecule database.

**Solution**:
```bash
# Either provide molecule_db_path
--molecule_db_path data/molecules.db

# Or disable molecular conditioning (not recommended)
--condition_on_molecule False
```

### Error: "Space group conditioning requires molecular conditioning"

**Cause**: Trying to use space group or density conditioning without molecular features.

**Solution**:
```bash
# Enable molecular conditioning
--condition_on_molecule True \
--molecule_db_path data/molecules.db
```

### NotImplementedError during sampling

**Cause**: Crystal sampling not yet integrated with diffusion model.

**Status**: Expected behavior. This will be implemented in Phase 7.

**Workaround**: None currently. Training and validation work, but analysis/sampling is pending.

## File Locations

```
train_test_crystal.py       # Main training functions
crystal/sampling.py          # Sampling functions (partial)
main_crystal.py             # Training script
tests/test_training_loop.py # Tests
PHASE6_TRAINING_LOOP_SUMMARY.md  # Full documentation
```

## Example Output

```
Epoch 1/10
Epoch: 0, iter: 0/100, Loss 1234.56, NLL: 1200.34, RegTerm: 34.2, GradNorm: 12.3
Epoch: 0, iter: 10/100, Loss 1100.23, NLL: 1080.12, RegTerm: 20.1, GradNorm: 8.5
...
Epoch time: 45.32s

Validating...
 Val NLL 	 epoch: 0, iter: 0/20, NLL: 1050.45
Validation NLL: 1050.45

Analyzing generated structures...
Sampling not yet implemented: Crystal sampling requires implementing sample()...
Validity: 0.00% (0/100)

Saving checkpoint to outputs/my_first_crystal/checkpoints/checkpoint_epoch0000.pt
```

## Next Steps

1. **Run tests**: Verify installation with `python tests/test_training_loop.py`
2. **Prepare data**: Create crystal and molecule ASE databases
3. **Start training**: Use minimum configuration first
4. **Monitor**: Check logs and checkpoints
5. **Phase 7**: Wait for sampling implementation or contribute!

## Support

For issues or questions:
1. Check `PHASE6_TRAINING_LOOP_SUMMARY.md` for detailed documentation
2. Review test files for usage examples
3. See `PROJECT_COMPLETE_SUMMARY.md` for overall architecture

## Contributing to Phase 7

If you want to implement the remaining features:

1. **Crystal Sampling**: 
   - Implement `sample()` in diffusion model
   - Handle cell parameter evolution
   - Test with `tests/test_training_loop.py`

2. **Loss Computation**:
   - Adapt `compute_loss_and_nll()` in `qm9/losses.py`
   - Add cell parameter loss term
   - Validate gradients

3. **Node Distribution**:
   - Fit distribution from crystal data
   - Implement in `main_crystal.py`
   - Update sampling calls

---

**Version**: 1.0.0  
**Date**: 2025-10-13  
**Status**: Phase 6 Complete, Phase 7 Pending
