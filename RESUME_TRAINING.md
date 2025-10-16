# Resume Training Guide

This guide explains how to resume training from a checkpoint in the E3 Diffusion for Molecules project.

## Quick Start

If you've trained a model and want to continue training from where it stopped:

```bash
python main_qm9.py \
    --exp_name your_experiment_name \
    --resume outputs/your_experiment_name \
    --n_epochs 500
```

That's it! The script will automatically:
- Load the model weights (prioritizing EMA version if available)
- Load the optimizer state
- Load training configuration
- Resume from the last saved epoch

## Problem Statement

After running training for 200 epochs, you want to continue training because the model needs more training. How do you resume from the checkpoint?

## Solution

### Original Training Command

```bash
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --model egnn_dynamics \
    --lr 1e-4 \
    --nf 256 \
    --n_layers 9 \
    --save_model True \
    --diffusion_steps 1000 \
    --sin_embedding False \
    --n_epochs 200 \
    --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --dequantization deterministic \
    --include_charges False \
    --diffusion_loss_type l2 \
    --batch_size 16 \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
    --dataset ase_db \
    --ase_db_path ase.db \
    --test_epochs 10 \
    --no_wandb
```

### Resume Training Command

```bash
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --resume outputs/exp_cond_molecular_descriptors \
    --n_epochs 500 \
    --no_wandb
```

This will:
1. Load all settings from the checkpoint
2. Resume from epoch 200 (automatically detected)
3. Continue training until epoch 500
4. Save results to `outputs/exp_cond_molecular_descriptors_resume`

## Features

### Automatic File Detection

The resume functionality automatically detects and loads:
1. **Model weights**: Tries `generative_model_ema.npy` first (EMA version), then `generative_model.npy`
2. **Optimizer state**: Loads `optim.npy` if available
3. **Training configuration**: Loads all parameters from `args.pickle`
4. **Start epoch**: Automatically determined from the checkpoint

### Flexible Resume Options

**Option 1: Resume from directory (Recommended)**
```bash
--resume outputs/exp_cond_molecular_descriptors
```

**Option 2: Resume from specific model file**
```bash
--resume outputs/exp_cond_molecular_descriptors/generative_model_ema.npy
```

**Option 3: Specify start epoch manually**
```bash
--resume outputs/exp_cond_molecular_descriptors --start_epoch 200
```

### Backward Compatibility

The implementation maintains backward compatibility with older checkpoints:
- Supports old naming convention (`flow.npy`)
- Works with checkpoints that don't have `current_epoch` saved
- Gracefully handles missing optimizer state

## Testing

To verify that the resume functionality works correctly:

```bash
python test_resume.py
```

This test script validates:
- Loading from checkpoint directory
- Loading from specific model file
- Backward compatibility with old format
- Existing checkpoint loading

## Helper Scripts

### Example Script

Run the example script to see detailed instructions:
```bash
./example_resume_training.sh
```

### Interactive Help

The script will:
1. Check if checkpoint exists
2. Verify required files
3. Show the resume command
4. Optionally run training

## Requirements

For resuming training, the checkpoint directory should contain:
- **Required**: `generative_model.npy` or `generative_model_ema.npy`
- **Recommended**: `optim.npy` (for optimal training continuation)
- **Recommended**: `args.pickle` (to preserve training configuration)

## Troubleshooting

### Error: "No model checkpoint found"

**Problem**: The checkpoint directory doesn't contain model files.

**Solution**: 
- Check that you used `--save_model True` during training
- Verify the checkpoint directory path is correct
- Check that training reached at least one test epoch (where models are saved)

### Warning: "Optimizer state not found"

**Problem**: `optim.npy` doesn't exist in the checkpoint.

**Impact**: Training will continue but with a fresh optimizer state. This might affect learning dynamics slightly.

**Solution**: This is just a warning. Training will continue normally. For best results, ensure `--save_model True` was used during original training.

### Training starts from epoch 0

**Problem**: `start_epoch` not properly detected from checkpoint.

**Solution**: 
- Check that `args.pickle` exists in the checkpoint directory
- The checkpoint was created with an older version that didn't save `current_epoch`
- Manually specify `--start_epoch` parameter

## Documentation

- **English**: See `doc/user_manual.md`, section "Resuming Training"
- **Japanese**: See `RESUME_TRAINING_JA.md`

## Technical Details

### What gets loaded from checkpoint?

From `args.pickle`:
- Model architecture (n_layers, nf, attention, etc.)
- Training parameters (lr, batch_size, etc.)
- Dataset configuration
- Conditioning settings
- Current epoch number

From `generative_model_ema.npy` or `generative_model.npy`:
- Model weights
- All trainable parameters

From `optim.npy`:
- Optimizer state
- Learning rate schedule state
- Momentum/adaptive learning rate parameters

### What can be overridden?

You can override:
- `--n_epochs`: Train for more epochs
- `--start_epoch`: Manually set starting epoch
- `--exp_name`: Use different experiment name
- `--no_wandb` / `--wandb_usr`: Change logging settings

You cannot override (loaded from checkpoint):
- Model architecture
- Dataset configuration
- Most hyperparameters

This ensures training consistency when resuming.

## Examples

### Continue training for more epochs
```bash
python main_qm9.py \
    --exp_name my_model \
    --resume outputs/my_model \
    --n_epochs 1000
```

### Resume with different logging
```bash
python main_qm9.py \
    --exp_name my_model \
    --resume outputs/my_model \
    --wandb_usr your_username
```

### Resume from specific epoch-numbered checkpoint
```bash
python main_qm9.py \
    --exp_name my_model \
    --resume outputs/my_model/generative_model_ema_190.npy \
    --start_epoch 190
```

## Notes

- The resumed experiment will be saved with suffix `_resume` (e.g., `exp_cond_molecular_descriptors_resume`)
- Checkpoints are saved only when validation loss improves
- Both numbered checkpoints (`generative_model_190.npy`) and best checkpoint (`generative_model.npy`) are saved
- EMA (Exponential Moving Average) models typically perform better than regular models

## Support

For issues or questions:
1. Check this guide
2. Read the documentation in `doc/user_manual.md`
3. Run `python test_resume.py` to verify functionality
4. Open an issue on GitHub with details about your setup
