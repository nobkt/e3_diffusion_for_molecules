# Numerical Stability Guide for E3 Diffusion Training

This guide addresses common numerical instability issues that can occur during training, particularly the "NaN in EGNN velocity output" warning.

## Problem Description

When training with certain parameter combinations, you may encounter warnings like:
```
Warning: detected nan in EGNN velocity output, resetting to zero.
  Velocity stats - min: -3873.587646, max: 3624.457275
  Input stats - h_final range: [nan, nan]
  This indicates numerical instability in the diffusion process.
```

## Root Causes

### 1. Extreme Diffusion Noise Precision
Using very small `diffusion_noise_precision` values (e.g., 1e-5) with polynomial schedules can create extreme log_SNR values (>±10), leading to numerical instability.

### 2. Normalization Factor Issues
Very small normalization factors in EGNN aggregation can cause division by zero or near-zero values.

### 3. Gradient Explosion
Extreme loss values can cause gradient explosion during backpropagation.

## Solutions Implemented

### 1. Robust Polynomial Schedule
- Automatically clamps precision parameter to minimum 1e-4
- Clips gamma values to [-10.0, 10.0] range
- Provides warnings when large gamma values are detected

### 2. Enhanced EGNN Stability
- NaN detection and replacement in inputs/outputs
- Safe normalization factor handling in `unsorted_segment_sum`
- Better error reporting with statistics

### 3. Training Loop Protection
- Extreme loss detection and batch skipping
- Gradient clipping (enabled by default)
- Better error handling during training

## Recommended Parameters

For stable training, use these parameter ranges:

### Conservative (Recommended for new datasets)
```bash
--diffusion_noise_precision 1e-4
--lr 1e-5
--batch_size 32
--diffusion_steps 1000
--normalization_factor 100
```

### Standard (QM9-tested)
```bash
--diffusion_noise_precision 1e-5
--lr 1e-4
--batch_size 64
--diffusion_steps 1000
--normalization_factor 1
```

### Aggressive (Use with caution)
```bash
--diffusion_noise_precision 1e-6
--lr 1e-4
--batch_size 128
--diffusion_steps 1000
--normalization_factor 1
```

## Warning Signs

Watch for these indicators of numerical instability:

1. **Large gamma values**: log_SNR range > ±8.0
2. **NaN warnings**: "detected nan in EGNN velocity output"
3. **Extreme losses**: Loss values > 1e6
4. **Gradient explosion**: Gradient norms > 100

## Troubleshooting Steps

If you encounter numerical instability:

1. **Increase diffusion_noise_precision**: Try 1e-4 instead of 1e-5
2. **Reduce learning rate**: Try 1e-5 instead of 1e-4
3. **Enable gradient clipping**: Use `--clip_grad True` (default)
4. **Increase normalization factor**: Try 100 instead of 1
5. **Reduce batch size**: Try 32 instead of 64

## Example: Fixing the Original Issue

The original problematic command:
```bash
python main_qm9.py --dataset ase_db --ase_db_path /path/to/ase.db \
    --exp_name molecular_descriptor_model_fr_pubchem --n_epochs 200 \
    --diffusion_steps 1000 --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \  # This was too small
    --include_charges False --diffusion_loss_type l2 --batch_size 32 \
    --model egnn_dynamics --lr 1e-5 --nf 256 --n_layers 9 --no_wandb
```

Recommended fix:
```bash
python main_qm9.py --dataset ase_db --ase_db_path /path/to/ase.db \
    --exp_name molecular_descriptor_model_fr_pubchem --n_epochs 200 \
    --diffusion_steps 1000 --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-4 \  # Increased for stability
    --include_charges False --diffusion_loss_type l2 --batch_size 32 \
    --model egnn_dynamics --lr 1e-5 --nf 256 --n_layers 9 --no_wandb \
    --normalization_factor 100  # Added for better numerical stability
```

## Technical Details

The fixes are implemented in:
- `equivariant_diffusion/en_diffusion.py`: Polynomial schedule stabilization
- `egnn/models.py`: EGNN NaN detection and handling
- `egnn/egnn_new.py`: Safe normalization in aggregation
- `train_test.py`: Training loop stability checks