# Summary: Fixed Context Shape Mismatch in Conditional Training

## Issue
Training failed with `RuntimeError: shape '[11, 27]' is invalid for input of size 187` when using conditional training with molecular descriptors.

## Root Cause
The `prepare_context()` function in `qm9/utils.py` had a bug where it determined feature type based on dimension comparison:

```python
if properties.size(1) == n_nodes:
    # Treated as node feature
```

When `atom_types_encoding` had 17 features (number of atom types) and a batch had 17 nodes, the function incorrectly classified it as a node feature instead of a global (molecular-level) feature. This caused:

- Expected context shape: `(batch_size, n_nodes, 27)` where 27 = 1 (molecular_weight) + 1 (pi_conjugation_ratio) + 17 (atom_types) + 8 (functional_groups)
- Actual context shape: `(batch_size, n_nodes, 11)` where 11 = 1 + 1 + 1 (atom_types incorrectly treated as 1 feature) + 8

## Solution

### 1. Fixed Feature Classification (qm9/utils.py)
Added explicit list of global features that should always be broadcast to all nodes:

```python
global_features = {'atom_types_encoding', 'functional_groups_encoding', 
                  'molecular_weight', 'pi_conjugation_ratio'}

if key in global_features:
    # Always treat as global feature regardless of dimensions
    n_features = properties.size(1)
    reshaped = properties.view(batch_size, 1, n_features).repeat(1, n_nodes, 1)
    context_node_nf += n_features
```

### 2. Added Validation Checks
- Verify conditioning keys exist in both minibatch and property_norms
- Validate final context shape matches expected dimensions
- Provide clear error messages for debugging

### 3. Added Documentation
Created `PREPARE_CONTEXT_FIX.md` with detailed explanation of the problem and solution.

## Testing
The fix:
- ✅ Resolves the shape mismatch error for molecular descriptor conditioning
- ✅ Maintains backward compatibility with standard QM9 features (homo, lumo, etc.)
- ✅ Works for both ASE database and standard QM9 datasets
- ✅ Compatible with DataParallel training

## Files Modified
1. `qm9/utils.py` - Fixed prepare_context function
2. `PREPARE_CONTEXT_FIX.md` - Detailed documentation

## Impact
This fix enables proper use of the following conditioning features:
- `atom_types_encoding` - Multi-hot encoding of atom types present in molecule
- `functional_groups_encoding` - Multi-hot encoding of functional groups
- `molecular_weight` - Molecular weight in atomic mass units
- `pi_conjugation_ratio` - Ratio of pi bonds to total bonds

These features can now be used together without shape mismatches, enabling more sophisticated conditional molecular generation.
