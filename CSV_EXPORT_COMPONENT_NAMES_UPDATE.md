# CSV Export Component Names Update

## Summary

The `main_qm9.py --export_training_stats` functionality has been updated to use specific component names instead of generic "Component_N" labels for `atom_types_encoding` and `functional_groups_encoding` exports.

## Changes Made

### 1. Updated `_export_encoding_statistics_csv` function
- Added `component_names` parameter to accept specific component names
- Modified CSV output to use provided names instead of generic "Component_i"
- Updated histogram file naming to use component names
- Sanitized component names for safe filename generation

### 2. Created `get_functional_group_smarts_patterns` function
- Returns the SMARTS patterns used for functional group encoding
- Patterns match those in `qm9/openbabel_functions.py`
- Maintains proper ordering for encoding consistency

### 3. Updated `export_training_statistics` function
- Added `dataset_info` parameter to access atom decoder and other metadata
- Modified calls to `_export_encoding_statistics_csv` to pass component names:
  - Atom types: Uses element names from `dataset_info['atom_decoder']`
  - Functional groups: Uses SMARTS patterns from `get_functional_group_smarts_patterns()`

### 4. Updated function call site
- Modified the call to `export_training_statistics` to include `dataset_info`

### 5. Updated test files and demo
- Modified `test_csv_export.py` to pass the new parameter
- Updated `demo_csv_export.py` to show the new format with specific names

## Output Changes

### Before (Generic Names)
```csv
Atom Type Component,Mean,Std,Min,Max,Q25,Q50,Q75,Non-zero Count,Non-zero %
Component_0,0.125000,0.235681,0.000000,0.800000,0.000000,0.000000,0.200000,95,57.23%
Component_1,0.650000,0.184521,0.200000,1.000000,0.500000,0.700000,0.800000,166,100.00%
```

### After (Specific Names)
```csv
Atom Type Component,Mean,Std,Min,Max,Q25,Q50,Q75,Non-zero Count,Non-zero %
H,0.125000,0.235681,0.000000,0.800000,0.000000,0.000000,0.200000,95,57.23%
C,0.650000,0.184521,0.200000,1.000000,0.500000,0.700000,0.800000,166,100.00%
```

### Functional Groups Example
```csv
Functional Group Component,Mean,Std,Min,Max,Q25,Q50,Q75,Non-zero Count,Non-zero %
[OH],0.045000,0.134521,0.000000,0.800000,0.000000,0.000000,0.000000,23,13.86%
[CX3]=[OX1],0.067000,0.156743,0.000000,0.600000,0.000000,0.000000,0.000000,34,20.48%
[CX3](=O)[OX2H1],0.032000,0.098765,0.000000,0.500000,0.000000,0.000000,0.000000,18,10.84%
```

## Functional Group SMARTS Patterns

The following 14 SMARTS patterns are used for functional group encoding:

1. `[OH]` - Hydroxyl (-OH)
2. `[CX3]=[OX1]` - Carbonyl (C=O)
3. `[CX3](=O)[OX2H1]` - Carboxyl (-COOH)
4. `[CX3H1](=O)[#6]` - Aldehyde (-CHO)
5. `[CX3](=O)([#6])[#6]` - Ketone (C=O)
6. `[NX3;H2,H1;!$(NC=O)]` - Amino (-NH2, -NH-)
7. `[N+](=O)[O-]` - Nitro (-NO2)
8. `[Cl]` - Chloro (-Cl)
9. `[Br]` - Bromo (-Br)
10. `[F]` - Fluoro (-F)
11. `[I]` - Iodo (-I)
12. `[CH3]` - Methyl (-CH3)
13. `[OX2]([#6])[CH3]` - Methoxy (-OCH3)
14. `c1ccccc1` - Phenyl (benzene ring)

## Files Modified

1. `main_qm9.py` - Core export functionality
2. `test_csv_export.py` - Test file parameter updates
3. `demo_csv_export.py` - Demo output format updates

## Backward Compatibility

The changes are backward compatible - if `component_names` is not provided, the function falls back to the original "Component_i" naming scheme.

## Usage

The functionality works with the same command as before:

```bash
python main_qm9.py --dataset ase_db --ase_db_path your_database.db --export_training_stats
```

The output CSV files will now contain meaningful component names that directly correspond to chemical entities (elements and functional groups).