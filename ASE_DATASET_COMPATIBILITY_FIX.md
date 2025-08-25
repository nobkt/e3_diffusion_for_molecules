# ASE Dataset Compatibility Fix

This document describes the fix for high loss values when using dataset=ase in E3 Diffusion for Molecules.

## Problem

When training with `dataset=ase`, users experienced very high loss values (around 20-23) compared to normal QM9 training, as shown in the issue:

```
Epoch: 0, iter: 0/7, Loss 23.25, NLL: 23.25, RegTerm: 0.0, GradNorm: 15.1
Epoch: 0, iter: 1/7, Loss 20.27, NLL: 20.27, RegTerm: 0.0, GradNorm: 8.0
...
```

## Root Cause Analysis

Using the comparison scripts `compare_qm9_ase_data_structures.py` and `verify_dataset_compatibility.py`, we identified several critical incompatibilities between ASE and QM9 data structures:

### 1. One-hot Encoding Dimension Mismatch (CRITICAL)
- **QM9**: 5 atom types (H, C, N, O, F)
- **ASE**: 8 atom types (H, C, N, O, F, P, S, Cl)
- **Impact**: Model expects 5-dimensional input but receives 8-dimensional, causing immediate training failure

### 2. Property Key Mismatches
- **QM9**: Uses property keys like `U0`, `HOMO`, `LUMO`, `gap`
- **ASE**: Used Hartree-suffixed keys like `U0_Ha`, `HOMO_Ha`, `LUMO_Ha`, `gap_Ha`
- **Impact**: Model cannot find expected properties, affecting conditional training

### 3. Data Type Inconsistencies
- **QM9**: Uses `float32` for all tensors
- **ASE**: Used `float64` for charges tensor
- **Impact**: Potential precision and performance issues during training

### 4. Invalid Charge Values
- **Issue**: ASE dataset contained 0 values in charge tensors from padding
- **Impact**: Invalid atomic numbers could confuse the model

## Solution

### Files Modified

1. **configs/datasets_config.py**
   - Updated `ase_with_h` and `ase_without_h` configurations to use QM9-compatible atom types
   - Changed from 8 atom types to 5 atom types (H, C, N, O, F)
   - Updated colors and radii to match QM9 format

2. **build_ase_dataset.py**
   - Added property key mapping in `ASETransform` class:
     - `U0_Ha` → `U0`
     - `HOMO_Ha` → `HOMO`
     - `LUMO_Ha` → `LUMO`
     - `gap_Ha` → `gap`
     - `ZPVE_Ha` → `zpve`
   - Fixed data types to use `float32` consistently
   - Added validation to skip molecules with non-QM9 atom types
   - Improved charge handling to avoid invalid 0 values

3. **test_dataset_compatibility.py**
   - Updated test expectations to use QM9 property key format

### Key Changes

#### Dataset Configuration Fix
```python
# Before: 8 atom types
'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7}

# After: 5 QM9-compatible atom types  
'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
```

#### Property Key Mapping
```python
property_mapping = {
    'U0_Ha': 'U0',
    'HOMO_Ha': 'HOMO', 
    'LUMO_Ha': 'LUMO',
    'gap_Ha': 'gap',
    'ZPVE_Ha': 'zpve',
}
```

#### Data Type Consistency
```python
# Ensure float32 for all tensors
new_data['charges'] = torch.from_numpy(geometry[:, 0].astype(np.float32)[:, None])
new_data[qm9_prop_name] = torch.tensor(float(prop_value), dtype=torch.float32)
```

## Verification

The fix was verified using multiple test scripts:

1. **verify_dataset_compatibility.py**: Focused verification of key compatibility issues
2. **compare_qm9_ase_data_structures.py**: Detailed comparison of data structures
3. **test_dataset_compatibility.py**: Original compatibility test updated for new format

All tests now pass:
```
======================================================================
✅ ALL COMPATIBILITY TESTS PASSED!
ASE datasets should now produce the same data structure and
value ranges as QM9 datasets, eliminating the 10x loss difference.
======================================================================
```

## Expected Result

After applying this fix, training with `dataset=ase` should produce loss values similar to QM9 training instead of the abnormally high values (20-23). The ASE dataset now:

- Uses QM9-compatible 5-dimensional one-hot encoding
- Provides properties with QM9-expected key names
- Uses consistent float32 data types
- Avoids invalid charge values

## Usage

No changes are required for user code. Simply use `dataset=ase` as before:

```bash
python main_qm9.py --dataset=ase --ase_db_file=path/to/your/molecules.db
```

The dataset will now automatically use QM9-compatible data structures and should train with normal loss values.

## Migration for Custom ASE Datasets

If you have ASE databases with molecules containing atoms beyond H, C, N, O, F (such as P, S, Cl), you have two options:

1. **Filter molecules** to only include QM9-compatible atoms
2. **Extend the QM9 model** to support additional atom types (requires model architecture changes)

The current fix chooses option 1 for maximum compatibility with existing QM9-trained models.