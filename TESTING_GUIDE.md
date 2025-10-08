# Testing Guide for Context Shape Fix

This guide explains how to test the fix for the context shape mismatch error.

## Quick Verification

Run the demonstration test to see the before/after comparison:

```bash
python3 test_context_fix.py
```

This test illustrates:
- The problematic scenario (17 nodes, 17 atom type features)
- Expected vs actual behavior before the fix
- Correct behavior after the fix

## Testing with Original Failing Command

The original error occurred with this command:

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

### Expected Behavior After Fix

1. **Training should proceed without shape mismatch errors**
   - No `RuntimeError: shape '[11, 27]' is invalid for input of size 187`
   - Context is correctly shaped as `(batch_size, n_nodes, 27)`

2. **Validation checks provide clear error messages if issues occur**
   - Missing conditioning keys are reported with available keys
   - Shape mismatches are caught with detailed diagnostics

## Testing Backward Compatibility

Test with standard QM9 conditioning features:

```bash
python main_qm9.py \
  --exp_name test_standard_qm9 \
  --model egnn_dynamics \
  --dataset qm9 \
  --conditioning homo lumo gap \
  --batch_size 32 \
  --n_epochs 1 \
  --no_wandb
```

Expected: Should work exactly as before, since homo/lumo/gap are 1D scalar features.

## Testing Mixed Conditioning

Test with mix of scalar and multi-dimensional features:

```bash
python main_qm9.py \
  --exp_name test_mixed_conditioning \
  --model egnn_dynamics \
  --dataset ase_db \
  --ase_db_path ase.db \
  --conditioning molecular_weight homo lumo atom_types_encoding \
  --batch_size 16 \
  --n_epochs 1 \
  --no_wandb
```

Expected: All features should be correctly processed regardless of dimensionality.

## Verifying the Fix in Code

The key changes in `qm9/utils.py`:

### 1. Explicit Global Feature List
```python
global_features = {'atom_types_encoding', 'functional_groups_encoding', 
                  'molecular_weight', 'pi_conjugation_ratio'}
```

### 2. Priority Check for Known Global Features
```python
if key in global_features:
    # Always treat as global feature
    n_features = properties.size(1)
    reshaped = properties.view(batch_size, 1, n_features).repeat(1, n_nodes, 1)
    context_node_nf += n_features
```

### 3. Validation Checks
```python
# Check keys exist
if key not in minibatch:
    raise ValueError(...)

# Validate final shape
if context.size(2) != context_node_nf:
    raise ValueError(...)
```

## Common Issues and Solutions

### Issue: Import errors or missing dependencies
**Solution**: Ensure torch is installed: `pip install torch`

### Issue: Database not found
**Solution**: Create or specify correct path with `--ase_db_path`

### Issue: Different conditioning features needed
**Solution**: The fix specifically handles:
- `atom_types_encoding` (multi-dimensional global)
- `functional_groups_encoding` (multi-dimensional global)
- `molecular_weight` (scalar global)
- `pi_conjugation_ratio` (scalar global)

Other features should be added to the `global_features` set if they exhibit similar behavior.

## Success Criteria

✅ Training completes multiple epochs without shape errors
✅ Context shape is `(batch_size, n_nodes, expected_features)` 
✅ Model can successfully reshape context for processing
✅ Standard QM9 features still work correctly
✅ Mixed conditioning works without issues

## Debugging

If issues persist, add debug output to `prepare_context`:

```python
print(f"Processing {key}: shape={properties.shape}, is_global={key in global_features}")
print(f"Context so far: node_nf={context_node_nf}, list_len={len(context_list)}")
```

This will help identify which feature is causing problems.
