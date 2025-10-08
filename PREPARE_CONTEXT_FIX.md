# Fix for prepare_context Shape Mismatch Error

## Problem
During conditional training with `--conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding`, 
the training would fail with:

```
RuntimeError: shape '[11, 27]' is invalid for input of size 187
```

This occurred at line 75 in `egnn/models.py`:
```python
context = context.view(bs*n_nodes, self.context_node_nf)
```

## Root Cause
The `prepare_context` function in `qm9/utils.py` had a bug where it would incorrectly classify features based on their dimensions:

```python
elif properties.size(1) == n_nodes:
    # Node feature with shape (batch_size, n_nodes)
    context_key = properties.unsqueeze(2)
    context_list.append(context_key)
    context_node_nf += 1  # Only adds 1 feature!
```

When `atom_types_encoding` had shape `(batch_size, n_atom_types)` where `n_atom_types == n_nodes` (e.g., both equal to 17),
the function would incorrectly treat it as a node feature instead of a global feature.

This caused:
- At initialization with dummy batch: `context_node_nf = 1 + 1 + 17 + 8 = 27` (correct)
- At runtime with different batch: `context_node_nf = 1 + 1 + 1 + 8 = 11` (incorrect when n_nodes = 17)
- Model expects 27 features but receives only 11, causing shape mismatch

## Solution
Added explicit classification of known global features:

```python
# Define which features are always global (molecular-level) features
# These should be broadcast to all nodes regardless of their dimensions
global_features = {'atom_types_encoding', 'functional_groups_encoding', 
                  'molecular_weight', 'pi_conjugation_ratio'}

for key in conditioning:
    if key in global_features:
        # Always treat as global feature and broadcast to all nodes
        n_features = properties.size(1)
        reshaped = properties.view(batch_size, 1, n_features).repeat(1, n_nodes, 1)
        context_list.append(reshaped)
        context_node_nf += n_features  # Adds all features correctly
```

This ensures:
- `atom_types_encoding` (batch_size, 17) → (batch_size, n_nodes, 17)
- `functional_groups_encoding` (batch_size, 8) → (batch_size, n_nodes, 8)
- `molecular_weight` (batch_size,) → (batch_size, n_nodes, 1)
- `pi_conjugation_ratio` (batch_size,) → (batch_size, n_nodes, 1)
- Total: context_node_nf = 1 + 1 + 17 + 8 = 27 ✓

## Additional Improvements
- Added validation checks to ensure conditioning keys exist in both minibatch and property_norms
- Added shape validation before returning context to catch issues early
- Improved error messages to help diagnose similar issues in the future

## Testing
The fix should be tested with:
```bash
python main_qm9.py --exp_name exp_cond_molecular_descriptors \
    --model egnn_dynamics --lr 1e-4 --nf 256 --n_layers 9 \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
    --dataset ase_db --ase_db_path ase.db \
    --batch_size 16 --include_charges False
```

The training should now proceed without shape mismatch errors.
