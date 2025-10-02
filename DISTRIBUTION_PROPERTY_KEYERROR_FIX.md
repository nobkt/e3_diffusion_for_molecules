# Fix for KeyError in DistributionProperty.sample()

## Problem Description

When running conditional training with the command:
```bash
python main_qm9.py --exp_name exp_cond_molecular_descriptors \
  --model egnn_dynamics --lr 1e-4 --nf 256 --n_layers 9 \
  --save_model True --diffusion_steps 1000 \
  --sin_embedding False --n_epochs 200 \
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
  --include_charges False \
  --no_wandb
```

The following error occurred during stability analysis after epoch 0:

```
Traceback (most recent call last):
  File "qm9/models.py", line 163, in sample
    dist = self.distributions[prop][n_nodes]
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
KeyError: 99
```

## Root Cause

The error occurs due to a mismatch between two distributions:

1. **DistributionNodes**: Created from the dataset's node count histogram (n_nodes), which includes all molecule sizes present in the dataset (e.g., 3, 5, 8 atoms).

2. **DistributionProperty**: Created from molecules that have the specified properties, but only stores distributions for node counts that actually have samples with those properties.

The problem happens when:
- The training dataset has a limited set of molecule sizes (e.g., only 3, 5, and 8 atoms)
- `nodes_dist.sample()` can return any of these sizes
- During sampling, if a node count is requested that doesn't exist in `prop_dist.distributions[prop]`, a KeyError occurs

In the reported case, the database `select.db` only contained molecules with 3, 5, and 8 atoms, but the code tried to sample properties for molecules with 99 atoms (or other sizes not in the training data).

## Solution

Modified `DistributionProperty.sample()` method in `qm9/models.py` to handle missing node counts gracefully:

```python
def sample(self, n_nodes=19):
    vals = []
    for prop in self.properties:
        # Handle missing node counts by finding the nearest available node count
        # This can happen when the node distribution includes counts that don't have
        # any training samples with the required properties
        if n_nodes not in self.distributions[prop]:
            # Find the nearest node count that exists in the distribution
            available_nodes = list(self.distributions[prop].keys())
            if len(available_nodes) == 0:
                raise ValueError(f"No distributions available for property {prop}")
            # Find the nearest node count
            n_nodes_actual = min(available_nodes, key=lambda x: abs(x - n_nodes))
        else:
            n_nodes_actual = n_nodes
        
        dist = self.distributions[prop][n_nodes_actual]
        idx = dist['probs'].sample((1,))
        val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
        val = self.normalize_tensor(val, prop)
        vals.append(val)
    vals = torch.cat(vals)
    return vals
```

The fix:
1. Checks if the requested `n_nodes` exists in the property distribution
2. If not, finds the nearest available node count
3. Uses the distribution for the nearest node count to sample the property value
4. This prevents KeyError while still providing reasonable property values

## Testing

Created comprehensive tests to verify the fix:

1. **Unit test** (`test_distribution_fix.py`): Tests the basic functionality with mock data
2. **Integration test** (`test_ase_db_fix.py`): Tests with actual ASE database loading
3. **Scenario simulation** (`test_exact_scenario.py`): Simulates the exact error scenario from the bug report

All tests pass successfully, confirming that:
- The KeyError no longer occurs
- Missing node counts are handled by using the nearest available distribution
- Batch sampling works correctly with mixed node counts
- Edge cases (very large or small node counts) are handled properly

## Impact

This fix allows conditional training to work with datasets that have:
- Limited molecule size ranges
- Sparse node count distributions
- ASE databases with specific molecule sizes

The fix is backward compatible and doesn't affect datasets where all node counts have property distributions (like standard QM9 dataset).
