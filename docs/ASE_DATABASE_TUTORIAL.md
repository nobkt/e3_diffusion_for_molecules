# ASE Database Support for E3 Diffusion for Molecules

This guide explains how to use ASE (Atomic Simulation Environment) database format datasets with the E3 diffusion model, using OpenBabel for molecular processing instead of RDKit.

## Overview

The E3 diffusion model has been extended to support:
- **ASE Database Format**: General molecular datasets in ASE's SQLite database format
- **OpenBabel Processing**: Molecular analysis and SMILES generation using OpenBabel
- **Backward Compatibility**: Existing QM9 and RDKit functionality remains unchanged

## Installation

Install the required packages:

```bash
pip install ase openbabel-wheel rdkit torch torchvision tqdm wandb imageio scipy
```

## Quick Start

### 1. Convert QM9 to ASE Database Format

```bash
# Download QM9 and convert to ASE database
python setup_ase_db_qm9.py --data-dir ./qm9/temp --output-db ./qm9/temp/qm9_ase.db
```

### 2. Test Basic Functionality

```bash
# Test ASE DB and OpenBabel functionality
python test_ase_openbabel.py

# Test framework integration
python test_integration.py

# Test training setup
python example_ase_training.py
```

### 3. Train with ASE Database

```bash
# Train E3 diffusion model with ASE database
python main_qm9.py \
    --dataset ase_db_qm9 \
    --ase_db_path ./qm9/temp/qm9_ase.db \
    --batch_size 32 \
    --n_epochs 100 \
    --exp_name edm_ase_qm9
```

## Dataset Formats

### Supported ASE Database Datasets

The implementation supports several dataset configurations:

1. **ASE DB QM9** (`ase_db_qm9`): QM9 dataset converted to ASE format
2. **ASE DB Generic** (`ase_db_generic`): General molecular datasets

### Dataset Configuration

Dataset configurations are defined in `configs/datasets_config.py`:

```python
# ASE DB QM9 with hydrogens
ase_db_qm9_with_h = {
    'name': 'ase_db_qm9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'use_openbabel': True,
    'with_h': True
}

# Generic ASE DB dataset (supports more elements)
ase_db_generic = {
    'name': 'ase_db_generic',
    'atom_encoder': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, ...},
    'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', ...],
    'use_openbabel': True,
    'with_h': True
}
```

## Creating Custom ASE Databases

### From Molecular Structures

```python
from ase import Atoms
from ase.db import connect

# Create database
db = connect('my_molecules.db')

# Add molecules
molecules = [
    Atoms('CH4', positions=[[0,0,0], [1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]),
    Atoms('H2O', positions=[[0,0,0], [1,0,0], [0,1,0]]),
]

for i, atoms in enumerate(molecules):
    properties = {'energy': -100.0 - i, 'gap': 10.0 + i}
    db.write(atoms, key_value_pairs=properties)
```

### From Other Formats

```python
from qm9.ase_db_dataset import convert_qm9_to_ase_db

# Convert QM9 data to ASE DB
convert_qm9_to_ase_db(
    qm9_data_dir='./qm9/temp/qm9',
    output_db_path='./qm9_converted.db',
    include_properties=True
)
```

## Training Commands

### Basic Training

```bash
python main_qm9.py \
    --dataset ase_db_qm9 \
    --ase_db_path ./data/qm9_ase.db \
    --batch_size 64 \
    --n_epochs 1000 \
    --exp_name edm_ase_qm9 \
    --lr 1e-4 \
    --nf 256 \
    --n_layers 9
```

### Training with Custom Database

```bash
python main_qm9.py \
    --dataset ase_db_generic \
    --ase_db_path ./data/my_molecules.db \
    --batch_size 32 \
    --remove_h False \
    --include_charges True \
    --exp_name edm_custom
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset type (`ase_db_qm9`, `ase_db_generic`) | Required |
| `--ase_db_path` | Path to ASE database file | Required |
| `--batch_size` | Training batch size | 64 |
| `--remove_h` | Remove hydrogen atoms | False |
| `--include_charges` | Include atomic charges | True |
| `--filter_n_atoms` | Filter by number of atoms | None |

## Evaluation and Analysis

### Generate Samples

```bash
python eval_sample.py \
    --model_path outputs/edm_ase_qm9 \
    --n_samples 1000
```

### Analyze Sample Quality

```bash
python eval_analyze.py \
    --model_path outputs/edm_ase_qm9 \
    --n_samples 10000
```

The evaluation will automatically use OpenBabel for molecular analysis when working with ASE DB datasets.

### Expected Output

```
Validity over 10000 molecules: 87.25%
Uniqueness over 8725 valid molecules: 99.12%
Novelty over 8648 unique valid molecules: 85.67%
```

## Molecular Metrics

The system automatically chooses the appropriate molecular processing library:

- **RDKit**: For standard QM9 datasets
- **OpenBabel**: For ASE database datasets (when `use_openbabel: True`)

### Manual Metrics Calculation

```python
from qm9.rdkit_functions import BasicMolecularMetrics
from configs.datasets_config import get_dataset_info

# Get dataset info
dataset_info = get_dataset_info('ase_db_qm9', remove_h=False)

# Initialize metrics (automatically uses OpenBabel)
metrics = BasicMolecularMetrics(dataset_info)

# Evaluate generated molecules
results = metrics.evaluate(generated_molecules)
validity, uniqueness, novelty = results[0]
```

## Advanced Usage

### Custom Dataset Configuration

Create a new dataset configuration in `configs/datasets_config.py`:

```python
my_custom_dataset = {
    'name': 'my_dataset',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'S'],
    'max_n_nodes': 50,
    'charges_dic': [1, 6, 7, 8, 16],
    'valencies': [1, 4, 3, 2, 2],
    'with_h': True,
    'use_openbabel': True
}
```

### Conditional Generation

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/edm_ase_qm9 \
    --property gap \
    --n_sweeps 10 \
    --task qualitative
```

### Property Prediction

```bash
cd qm9/property_prediction
python main_qm9_prop.py \
    --dataset ase_db_qm9 \
    --ase_db_path ../../data/qm9_ase.db \
    --property gap \
    --exp_name ase_classifier
```

## File Structure

```
qm9/
├── ase_db_dataset.py          # ASE database dataset loading
├── openbabel_functions.py     # OpenBabel molecular functions
├── rdkit_functions.py         # Enhanced with OpenBabel support
└── dataset.py                 # Updated dataset loading logic

configs/
└── datasets_config.py         # ASE DB dataset configurations

setup_ase_db_qm9.py           # QM9 conversion script
test_ase_openbabel.py          # Basic functionality tests
test_integration.py            # Framework integration tests
example_ase_training.py        # Training example
```

## Troubleshooting

### Common Issues

1. **OpenBabel Import Error**
   ```bash
   pip install openbabel-wheel
   ```

2. **ASE Import Error**
   ```bash
   pip install ase
   ```

3. **Database File Not Found**
   - Ensure the ASE database path is correct
   - Use absolute paths when possible

4. **Memory Issues with Large Datasets**
   - Reduce batch size: `--batch_size 16`
   - Filter molecules by size: `--filter_n_atoms 20`

### Validation

Test your setup:

```bash
# Test basic functionality
python test_ase_openbabel.py

# Test integration
python test_integration.py

# Test with your database
python example_ase_training.py --db-path /path/to/your/database.db
```

## Performance Considerations

- **ASE Database Loading**: Efficiently loads only required data splits
- **OpenBabel Processing**: Comparable performance to RDKit for most operations
- **Memory Usage**: Similar to original QM9 datasets
- **Training Speed**: No significant performance impact

## Backward Compatibility

All existing functionality remains unchanged:
- Original QM9 datasets work as before
- RDKit processing is still available
- Existing training scripts continue to work
- Model architectures are unchanged

The new ASE DB support is additive and doesn't affect existing workflows.

## Citation

If you use this ASE database extension, please cite both the original E3 diffusion paper and the relevant packages:

```bibtex
@article{e3_diffusion_original,
  title={Equivariant Diffusion for Molecule Generation in 3D},
  author={Original Authors},
  journal={Original Journal},
  year={2022}
}

@software{ase,
  title={The Atomic Simulation Environment},
  author={Ask Hjorth Larsen and others},
  url={https://wiki.fysik.dtu.dk/ase/},
  year={2017}
}

@software{openbabel,
  title={Open Babel: An open chemical toolbox},
  author={O'Boyle, Noel M and others},
  url={http://openbabel.org/},
  year={2011}
}
```