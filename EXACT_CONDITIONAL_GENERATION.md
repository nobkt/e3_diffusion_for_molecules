# Exact Conditional Molecular Generation

This document describes the new exact conditional generation functionality that allows specifying precise molecular descriptor conditions for molecule generation.

## Overview

The system now supports specifying exact conditions like:
```
molecular_weight=50.0 pi_conjugation_ratio=0.9 atom_types_encoding=[C,H,N,O] functional_groups_encoding=[[CX3](=O)[OX2H1],[NX3;H2,H1;!$(NC=O)]]
```

Instead of generating sweeps from minimum to maximum property values, you can now generate molecules that match specific criteria exactly.

## Quick Start

### 1. Train a Model with Molecular Descriptor Conditioning

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path your_database.db \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
    --exp_name molecular_descriptor_model \
    --n_epochs 1000 \
    --batch_size 32 \
    --lr 1e-4 \
    --nf 192 \
    --n_layers 9
```

### 2. Generate Molecules with Exact Conditions

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]' \
    --n_sweeps 5
```

## Supported Molecular Descriptors

### 1. Molecular Weight (`molecular_weight`)
- **Type**: Scalar (float)
- **Units**: Atomic mass units (u)
- **Example**: `molecular_weight=50.0`
- **Description**: Total molecular weight calculated from atomic masses

### 2. π Conjugation Ratio (`pi_conjugation_ratio`)
- **Type**: Scalar (float, 0.0-1.0)
- **Example**: `pi_conjugation_ratio=0.9`
- **Description**: Ratio of π bonds (double/aromatic) to total bonds

### 3. Atom Types Encoding (`atom_types_encoding`)
- **Type**: List of atomic symbols
- **Example**: `atom_types_encoding=[C,H,N,O]`
- **Description**: Specifies which atom types should be present in generated molecules

### 4. Functional Groups Encoding (`functional_groups_encoding`)
- **Type**: List of functional group names
- **Example**: `functional_groups_encoding=[carbonyl,hydroxyl,amino]`
- **Description**: Specifies which functional groups should be present

## Command Line Interface

### New Arguments

- `--use_exact_conditions`: Enable exact conditional generation
- `--property_values`: Specify exact property values (see format below)

### Property Values Format

The `--property_values` argument accepts a comma-separated list of `property=value` pairs:

```bash
# Single property
--property_values 'molecular_weight=50.0'

# Multiple scalar properties
--property_values 'molecular_weight=50.0,pi_conjugation_ratio=0.9'

# Including list properties
--property_values 'molecular_weight=50.0,atom_types_encoding=[C,H,N,O]'

# All descriptors (problem statement example)
--property_values 'molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O],functional_groups_encoding=[carbonyl,amino]'
```

### Format Rules

- Use commas to separate different properties
- Use equals signs to separate property names from values
- Use square brackets for lists: `[C,H,N,O]`
- No spaces around equals signs in property specifications
- Float values can include decimal points: `50.0` or `50`

## Examples

### Example 1: Generate Small Organic Molecules
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/your_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=30.0,atom_types_encoding=[C,H,O]' \
    --n_sweeps 3
```

### Example 2: Generate Highly Conjugated Molecules
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/your_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,N]' \
    --n_sweeps 3
```

### Example 3: Generate Molecules with Specific Functional Groups
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/your_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'functional_groups_encoding=[carbonyl,hydroxyl],molecular_weight=60.0' \
    --n_sweeps 3
```

### Example 4: Problem Statement Example
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/your_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]' \
    --n_sweeps 5
```

## Implementation Details

### Property Value Parsing
- Robust parsing handles complex nested structures
- Automatic type detection (float, list, string)
- Support for special characters in functional group specifications

### Context Tensor Creation
- Exact values are normalized using training statistics
- Multi-dimensional properties are properly encoded
- Binary encodings for categorical properties (atom types, functional groups)

### Sampling Integration
- New `sample_exact_conditional` function
- Seamless integration with existing sampling infrastructure
- Maintains compatibility with original sweep-based generation

### Backwards Compatibility
- Original sweep-based generation still available (default behavior)
- All existing functionality preserved
- New features activated only with `--use_exact_conditions` flag

## Testing

Run the test suite to validate the implementation:

```bash
# Test property parsing
python test_exact_conditions.py

# Test context creation
python test_exact_context.py

# View demonstrations
python exact_conditioning_demo.py
```

## Molecular Descriptor Extraction

The system automatically extracts molecular descriptors during training using:

- **ASE (Atomic Simulation Environment)**: For atomic properties and molecular weight
- **OpenBabel**: For functional group detection and π bond analysis
- **Custom algorithms**: For atom type encoding and property normalization

## File Structure

```
├── eval_conditional_qm9.py          # Main evaluation script (modified)
├── qm9/sampling.py                  # Sampling functions (new exact sampling)
├── qm9/openbabel_functions.py       # Molecular descriptor extraction
├── test_exact_conditions.py         # Test property parsing
├── test_exact_context.py           # Test context creation
├── exact_conditioning_demo.py       # Comprehensive demonstration
└── EXACT_CONDITIONAL_GENERATION.md # This documentation
```

## Troubleshooting

### Common Issues

1. **"Property X was not used for conditioning during training"**
   - Ensure your model was trained with the desired properties in `--conditioning`
   - Check that property names match exactly

2. **"Failed to parse property values"**
   - Check format: `property=value,property2=value2`
   - Ensure no spaces around equals signs
   - Use square brackets for lists: `[C,H,N,O]`

3. **"Context tensor dimension mismatch"**
   - Ensure the number of conditioning properties matches training
   - Check that list properties have consistent dimensions

### Dependencies

Required packages:
- `torch`
- `ase` (Atomic Simulation Environment)
- `openbabel-wheel` (for functional group detection)
- `numpy`

## Performance Notes

- Exact conditional generation has similar computational cost to sweep generation
- Context tensor creation is efficient and cached
- Memory usage scales with the number of conditioning properties
- Normalization uses pre-computed statistics from training data

## Future Enhancements

- Support for more complex functional group specifications (SMARTS patterns)
- Interactive property value selection
- Real-time property validation during generation
- Integration with molecular property prediction models

---

This implementation fulfills the problem statement requirements for exact conditional molecular generation with support for `molecular_weight`, `pi_conjugation_ratio`, `atom_types_encoding`, and `functional_groups_encoding` properties.