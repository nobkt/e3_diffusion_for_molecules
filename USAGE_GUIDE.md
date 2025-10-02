# How to Use: Fixed Conditional Training with ASE Databases

## Overview
The KeyError issue in conditional training with ASE databases has been fixed. You can now run the original command without errors.

## Fixed Command
The following command now works correctly:

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
  --ase_db_path select.db \
  --test_epochs 10 \
  --no_wandb
```

## What Was Fixed
The `DistributionProperty.sample()` method in `qm9/models.py` now handles cases where:
- Your training dataset has molecules with limited size ranges (e.g., only 3, 5, 8 atoms)
- The code needs to sample properties for molecule sizes not present in training data

Instead of raising a KeyError, it now:
1. Finds the nearest available molecule size in the training data
2. Uses that distribution to sample property values
3. Continues training without errors

## Expected Behavior
After the fix:
- ✓ Training starts normally
- ✓ Epoch 0 completes successfully
- ✓ Stability analysis runs without KeyError
- ✓ Training continues for all epochs

## Technical Details
For detailed technical information about the fix, see:
- `DISTRIBUTION_PROPERTY_KEYERROR_FIX.md` - Comprehensive documentation
- Commit history in this branch

## Notes
- The fix is backward compatible with existing datasets
- Works with any ASE database, regardless of molecule size distribution
- No changes needed to your command or dataset
- Property values are sampled from the nearest available molecule size distribution
