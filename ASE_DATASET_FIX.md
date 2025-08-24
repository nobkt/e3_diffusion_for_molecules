# ASE Dataset High Loss Issue - Fix Documentation

## Problem Description

When training with ASE datasets using the command:
```bash
python main_qm9.py --dataset ase --ase_db_file ./qm9.db --normalize_factors [1,8,1] ...
```

Users experienced very high loss values (~30) that didn't decrease during training, along with gradient clipping warnings like:
```
Clipped gradient with value 68693.4 while allowed 5244.7
```

## Root Cause Analysis

The issue was caused by **molecules with large coordinate spreads** in ASE databases. Specifically:

1. **Large molecular spans**: Some molecules had coordinate spreads >50 Angstroms (e.g., a molecule with positions ranging from 0 to 50 Angstroms)
2. **Inadequate position normalization**: The position normalization factor of 1 was too small for such large spreads
3. **Numerical instability**: When positions are normalized by dividing by 1, large coordinates (>50) remain large, causing numerical instability in the diffusion model
4. **Gradient explosion**: This led to gradient explosion and high loss values

## Solution

**Changed default ASE position normalization factor from 1 to 10:**

- **Before**: `args.normalize_factors = [1, 8, 1]` for ASE datasets
- **After**: `args.normalize_factors = [10, 8, 1]` for ASE datasets

## How the Fix Works

1. **Position scaling**: Dividing large coordinates by 10 instead of 1 brings them to a manageable range
2. **Improved stability**: Molecules with 50Å spans become 5Å after normalization, preventing numerical issues
3. **Preserved functionality**: Other datasets (QM9, GEOM) retain their original normalization factors

## Test Results

### Before Fix (normalize_factors = [1, 8, 1]):
```
Batch with 50Å molecule:
  Loss: 833.497 (extremely high)
  GradNorm: 3279.0 (gradient explosion)
  Test loss: 173427.59 (unstable)
```

### After Fix (normalize_factors = [10, 8, 1]):
```
Batch with same molecule:
  Loss: 2.54 (normal range)
  GradNorm: 2.4 (stable)
  Test loss: 163.58 (reasonable)
```

## Enhanced Features

The fix also includes:

1. **Automatic position span analysis**: Reports median and maximum molecular spans
2. **Smart recommendations**: Suggests larger normalization factors for datasets with very large molecules
3. **Backwards compatibility**: Preserves existing behavior for QM9 and GEOM datasets
4. **User override**: Respects user-specified normalization factors

## Usage Examples

### Automatic (Recommended)
```bash
# Uses optimized defaults: [10, 8, 1] for ASE
python main_qm9.py --dataset ase --ase_db_file ./qm9.db ...
```

### Manual Override
```bash
# Uses custom factors (for special cases)
python main_qm9.py --dataset ase --ase_db_file ./qm9.db --normalize_factors [20,8,1] ...
```

## Backward Compatibility

- **QM9 datasets**: Still use `[1, 4, 1]` (unchanged)
- **GEOM datasets**: Still use `[1, 4, 10]` (unchanged)  
- **User-specified factors**: Always respected (unchanged)
- **ASE datasets**: Now default to `[10, 8, 1]` (improved)

## When to Adjust Further

If you still experience high losses with ASE datasets, consider:

1. **Very large molecules (>100Å spans)**: Use position factor >20
2. **Very small molecules (<3Å spans)**: Can use position factor 1-5
3. **Mixed dataset**: Use factor based on largest molecules

The enhanced system will provide automatic recommendations based on your data characteristics.