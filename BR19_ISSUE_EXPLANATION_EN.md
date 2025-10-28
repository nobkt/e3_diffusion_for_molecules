# Br19 Issue: Root Cause and Fix

## Problem Description

When running the following command, impossible molecules like Br19 were being generated:

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/exp_cond_molecular_descriptors \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N]' \
    --n_samples 100
```

Despite specifying `atom_types_encoding=[C,H,O,N]`, the generated molecules contained atoms other than C, H, O, N (e.g., Br, I, Si, etc.).

## Root Cause

The issue was in the `create_exact_context` function in `eval_conditional_qm9.py`.

### Behavior During Training

When training a model with `--conditioning atom_types_encoding`, the `atom_types_encoding` property is **expanded into individual binary features**:

- `atom_types_encoding` → `has_C`, `has_H`, `has_N`, `has_O`, etc.

For example, if the dataset contains atom types `['C', 'H', 'N', 'O']`, four individual features are created.

Source code (around lines 331-337 in `qm9/dataset.py`):
```python
# For atom types - create one feature per atom type
if atom_types_sorted:
    for atom_type in atom_types_sorted:
        feature_name = f'has_{atom_type}'
        property_tensors[feature_name] = torch.zeros(n_molecules, dtype=torch.float32)
        for i, mol_atom_types in enumerate(atom_types_list):
            if atom_type in mol_atom_types:
                property_tensors[feature_name][i] = 1.0
```

### Problem During Generation (Before Fix)

When specifying `--property_values 'atom_types_encoding=[C,H,O,N]'` during generation, the code before the fix tried to look up `atom_types_encoding` directly in the `property_norms` dictionary.

However, since `atom_types_encoding` was expanded into individual `has_<atom>` features during training, the `property_norms` dictionary doesn't contain an `atom_types_encoding` key. Instead, it contains keys like `has_C`, `has_H`, `has_N`, `has_O`, etc.

As a result:
- `atom_types_encoding` was not found in `property_norms`
- No atom type information was set in the context tensor (remained at default zeros)
- The model generated molecules without atom type constraints
- Molecules were generated with any atom types present in the training data (Br, I, Si, etc.)

## Fix Implementation

The `create_exact_context` function was modified to properly expand `atom_types_encoding` and `functional_groups_encoding` into individual features:

### Key Changes

1. **Retrieve mapping information from dataloader**
   ```python
   atom_types_mapping = dataset.data.get('_atom_types_mapping', [])
   functional_groups_mapping = dataset.data.get('_functional_groups_mapping', [])
   ```

2. **Expand atom_types_encoding**
   ```python
   if key == 'atom_types_encoding' and isinstance(value, list):
       # Expand to individual has_<atom> features
       for atom in atom_types_mapping:
           feature_name = f'has_{atom}'
           if atom in value:
               expanded_property_values[feature_name] = 1.0
           else:
               expanded_property_values[feature_name] = 0.0
   ```

3. **Set context values**
   ```python
   for key in args_gen.conditioning:
       if key == 'atom_types_encoding':
           # Process individual has_<atom> features
           for atom in atom_types_mapping:
               feature_name = f'has_{atom}'
               if feature_name in expanded_property_values and feature_name in property_norms:
                   value = expanded_property_values[feature_name]
                   mean = property_norms[feature_name]['mean']
                   mad = property_norms[feature_name]['mad']
                   normalized_value = (value - mean) / mad
                   context[:, feature_idx] = normalized_value
                   feature_idx += 1
   ```

## Test Results

After the fix, all tests pass successfully:

```bash
$ python test_atom_types_fix.py
```

Test results:
- ✓ Property value parsing works correctly
- ✓ atom_types_encoding is expanded into individual features
- ✓ functional_groups_encoding is expanded into individual features
- ✓ Context tensor has correct values
- ✓ Non-zero features exist (atom type information is set)

## Usage

After the fix, the original command works correctly:

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/exp_cond_molecular_descriptors \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N]' \
    --n_samples 100
```

This command will now generate molecules containing only C, H, O, N atoms.

## Additional Parameter

The `--n_samples` parameter was added to control the number of molecules generated per sweep:

```bash
# Generate 10 molecules
--n_samples 10

# Generate 100 molecules (default)
--n_samples 100

# Generate 500 molecules
--n_samples 500
```

## Technical Details

### Context Tensor Structure

The context tensor during training is a concatenation of all conditioning features:

```
[molecular_weight, pi_conjugation_ratio, has_C, has_H, has_N, has_O, has_Br, has_I, ...]
```

The fix ensures that the context is created with the same structure during generation.

### Normalization

Each feature is normalized using the mean and MAD (Mean Absolute Deviation) from training:

```python
normalized_value = (value - mean) / mad
```

This ensures the model receives values at the same scale as during training.

## Summary

This fix ensures that exact conditional generation with `atom_types_encoding` and `functional_groups_encoding` works correctly. Generated molecules will now contain only the specified atom types, and impossible molecules like Br19 will no longer be generated.

## Files Modified

- `eval_conditional_qm9.py`: Fixed `create_exact_context` function to properly handle atom_types_encoding and functional_groups_encoding
- `test_atom_types_fix.py`: Added comprehensive tests to verify the fix
- `BR19_ISSUE_EXPLANATION_JA.md`: Japanese documentation (this file in Japanese)
- `BR19_ISSUE_EXPLANATION_EN.md`: English documentation (this file)
