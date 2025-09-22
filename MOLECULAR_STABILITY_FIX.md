# Molecular Stability Issue Fix Summary

## Problem Solved

You reported 0% molecular stability during training with the ASE database, with warnings about very short (<0.8 Å) and very long (>5.0 Å) distances.

## Root Cause Identified

The issue was in `main_qm9.py` where your explicit `--normalize_factors 3.0 4 1` argument was being silently overridden when using ASE databases. The code was designed to automatically compute normalization factors for ASE data, but it ignored user-provided values.

## Fix Implemented

### 1. Respect User's Explicit Choice
- Your `--normalize_factors 3.0 4 1` will now be used as intended
- Override only happens if you use default values `[1, 4, 1]`
- Clear messages explain which factors are being used

### 2. Enhanced Validation
- Warnings for normalization factors outside recommended range (0.5-10.0)
- Troubleshooting messages for extreme distance issues
- Better guidance for ASE database users

### 3. Improved Feedback
The code now prints clear messages like:
```
✅ Using user-specified normalize_factors: [3.0, 4.0, 1.0]
Note: For ASE databases, coordinate normalization factor was computed but overridden by user
✅ Coordinate normalization factor 3.0 is within recommended range
```

## How to Use

Your original command should now work correctly:
```bash
python main_qm9.py --dataset ase_db --ase_db_path /path/to/ase.db \
  --exp_name molecular_descriptor_model_fr_pubchem \
  --n_epochs 200 --diffusion_steps 1000 --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 \
  --include_charges False --diffusion_loss_type l2 --batch_size 32 \
  --model egnn_dynamics --lr 5e-5 --nf 256 --n_layers 9 --no_wandb \
  --normalize_factors 3.0 4 1
```

## Expected Improvement

With the fix:
- Your normalization factors `[3.0, 4, 1]` will be respected
- Coordinates should be properly scaled during training and sampling
- Molecular stability should improve significantly (target: >30%, ideally >70%)
- Distance warnings should be reduced or eliminated

## Additional Recommendations

1. **Monitor the output**: Look for the new messages confirming your factors are being used
2. **Check stability**: The molecular stability should improve dramatically
3. **Validate coordinates**: Distances should be more reasonable (1-4 Å typical for bonds)
4. **If issues persist**: The new troubleshooting messages will help identify remaining problems

## Normalization Factor Guidelines

- **Coordinate factor (first value)**: 1.5-5.0 typical for molecular systems
- **Your value 3.0**: Well within recommended range, should work well
- **Too low (<0.5)**: Can cause numerical instabilities
- **Too high (>10.0)**: Can lead to poor diffusion dynamics

The fix ensures your explicit choices are respected while providing guidance for optimal values.