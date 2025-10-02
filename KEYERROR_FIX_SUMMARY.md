# KeyError Fix Summary

## Problem Description

When training with conditional generation on an ASE database using the following command:

```bash
python main_qm9.py --exp_name exp_cond_molecular_descriptors \
  --model egnn_dynamics --lr 1e-4 --nf 256 --n_layers 9 \
  --save_model True --diffusion_steps 1000 --sin_embedding False \
  --n_epochs 200 --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 \
  --dequantization deterministic --include_charges False \
  --diffusion_loss_type l2 --batch_size 16 \
  --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
  --dataset ase_db --ase_db_path select.db --test_epochs 10 \
  --include_charges False --no_wandb
```

The training would fail at epoch 10 with:

```
KeyError: 19
```

The error occurred in `qm9/sampling.py` at line 217:
```python
min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
```

## Root Cause

The `sample_sweep_conditional` function in `qm9/sampling.py` was hardcoded to use `n_nodes=19` (the default from QM9 dataset) when generating conditional samples. However, when using a custom ASE database like `select.db`, the database might not contain any molecules with exactly 19 nodes.

For example, `select.db` only contains molecules with 3, 5, and 8 nodes. When the code tried to access `prop_dist.distributions[key][19]`, it raised a `KeyError` because that key doesn't exist in the distribution dictionary.

## Solution

The fix adds logic to handle missing node counts by finding the nearest available node count, similar to how the `DistributionProperty.sample()` method already handles this case.

### Changes Made

**File: `qm9/sampling.py`**

In the `sample_sweep_conditional` function, added a check before accessing the distribution:

```python
# Handle missing node counts by finding the nearest available node count
if n_nodes not in prop_dist.distributions[key]:
    available_nodes = list(prop_dist.distributions[key].keys())
    if len(available_nodes) == 0:
        raise ValueError(f"No distributions available for property {key}")
    n_nodes_actual = min(available_nodes, key=lambda x: abs(x - n_nodes))
else:
    n_nodes_actual = n_nodes

min_val, max_val = prop_dist.distributions[key][n_nodes_actual]['params']
```

### How It Works

1. **Check if requested node count exists**: Before accessing the distribution, check if `n_nodes` is a key in the distribution dictionary
2. **Find nearest available**: If not found, get all available node counts and find the one closest to the requested count using `min()` with a distance-based key function
3. **Use the nearest count**: Use the nearest available node count to get the distribution parameters

### Example Behavior

Given a database with molecules having 3, 5, and 8 nodes:

- Request n_nodes=19 → Uses n_nodes=8 (nearest available)
- Request n_nodes=5 → Uses n_nodes=5 (exact match)
- Request n_nodes=4 → Uses n_nodes=3 or 5 (both equidistant)
- Request n_nodes=1 → Uses n_nodes=3 (nearest available)

## Testing

Created two test files to verify the fix:

### 1. Unit Test: `test_keyerror_fix.py`

Tests the core logic in isolation:
- ✅ Missing node count correctly falls back to nearest
- ✅ Existing node count uses exact match
- ✅ Context tensor is created correctly

### 2. Integration Test: `test_integration_keyerror.py`

Tests the full workflow:
- ✅ Loads ASE database with sparse node counts
- ✅ Creates property distributions
- ✅ Generates conditional samples without KeyError
- ✅ Produces valid molecular structures

Both tests pass successfully.

## Benefits

1. **Robustness**: Works with any ASE database regardless of its node count distribution
2. **Backward Compatible**: Existing behavior is preserved when exact node count exists
3. **Consistent**: Uses the same fallback strategy as `DistributionProperty.sample()`
4. **Minimal Change**: Only adds a few lines of defensive code

## Usage

After this fix, the original command now works successfully:

```bash
python main_qm9.py --exp_name exp_cond_molecular_descriptors \
  --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
  --dataset ase_db --ase_db_path select.db \
  [other arguments...]
```

The training will complete without KeyError, even when the database doesn't contain molecules with the default node count (19).
