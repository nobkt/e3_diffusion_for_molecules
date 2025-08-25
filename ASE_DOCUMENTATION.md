# ASE Database Support for E3 Diffusion

This document explains how to use the ASE database support that has been added to the E3 diffusion model for molecules.

## Overview

The repository has been extended to support molecular datasets from ASE (Atomic Simulation Environment) databases, with OpenBabel used for molecular processing instead of RDKit when needed. This allows the model to work with any molecular dataset that can be stored in ASE database format.

## Key Features

- **ASE Database Support**: Load molecular datasets from ASE database files (.db)
- **OpenBabel Integration**: Use OpenBabel for molecular processing when RDKit is not available
- **Automatic Dataset Configuration**: Automatically determine atom types, encodings, and dataset statistics
- **Flexible Data Splitting**: Create train/validation/test splits from ASE databases
- **Full Pipeline Compatibility**: Works with all existing training, evaluation, and analysis functions

## Installation

The following additional dependencies are required:

```bash
pip install ase openbabel-wheel
```

These have been added to `requirements.txt`.

## Usage

### 1. Creating an ASE Database

You can create an ASE database from molecular data in several ways:

#### From QM9 Data (for testing)
```python
from qm9.ase_database import create_ase_database_from_qm9

# Convert QM9 npz file to ASE database
create_ase_database_from_qm9('path/to/qm9_train.npz', 'qm9_train.db', 'train')
```

#### From Individual Molecules
```python
from ase import Atoms
from ase.db import connect

db = connect('my_molecules.db')

# Add molecules one by one
mol = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
db.write(mol, energy=-1.0, custom_prop='water')

mol = Atoms('CH4', positions=[[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], 
                              [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]])
db.write(mol, energy=-5.2, custom_prop='methane')
```

### 2. Loading ASE Database for Training

#### Using the ASE Dataset Loader
```python
from qm9 import dataset

# Configure for ASE database
class Config:
    def __init__(self):
        self.dataset = 'ase_molecules'
        self.ase_db_path = 'path/to/molecules.db'
        self.batch_size = 32
        self.include_charges = True
        self.remove_h = False
        self.datadir = './temp'
        self.num_workers = 0

cfg = Config()
dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)
```

#### Using the ASE Reader Directly
```python
from qm9.ase_database import ASEDatabaseReader

# Initialize reader
ase_reader = ASEDatabaseReader('path/to/molecules.db')

# Get dataset info
dataset_info = ase_reader.get_dataset_info('my_dataset', with_h=True)

# Create splits
data_splits = ase_reader.create_splits(
    train_ratio=0.8, 
    valid_ratio=0.1, 
    test_ratio=0.1
)
```

### 3. Training with ASE Databases

Use the provided `main_ase.py` script:

```bash
python main_ase.py \
    --dataset ase_molecules \
    --ase_db_path path/to/molecules.db \
    --exp_name my_ase_experiment \
    --n_epochs 10 \
    --batch_size 32
```

### 4. Using OpenBabel Functions

The OpenBabel functions provide alternatives to RDKit functionality:

```python
from qm9.openbabel_functions import OpenBabelMolecularMetrics, xyz_to_smiles_openbabel

# Convert molecular positions to SMILES
positions = np.array([[0, 0, 0], [1, 0, 0]])  # Example positions
atom_types = [0, 1]  # C, H indices
atom_decoder = ['C', 'H']

smiles = xyz_to_smiles_openbabel(positions, atom_types, atom_decoder)

# Calculate molecular metrics
dataset_info = {'atom_decoder': ['C', 'H'], 'name': 'test'}
metrics = OpenBabelMolecularMetrics(dataset_info)

molecule_list = [(positions, atom_types)]  # List of (positions, atom_types) tuples
validity, uniqueness, novelty = metrics.evaluate(molecule_list)
```

## Dataset Configuration

ASE databases automatically generate dataset configurations. The configuration includes:

- `atom_encoder`: Mapping from atom symbols to indices
- `atom_decoder`: List of atom symbols
- `max_n_nodes`: Maximum number of atoms in any molecule
- `n_nodes`: Distribution of molecule sizes
- `atom_types`: Count of each atom type
- `colors_dic`: Colors for visualization
- `radius_dic`: Atomic radii for visualization

## Molecular Analysis

The molecular analysis functions work with both RDKit and OpenBabel:

```python
from qm9.analyze import analyze_stability_for_molecules

# Prepare molecule data
molecule_list = {
    'x': positions_tensor,
    'one_hot': one_hot_tensor,
    'node_mask': mask_tensor
}

# Analyze molecules
validity_dict, molecular_metrics = analyze_stability_for_molecules(molecule_list, dataset_info)

print(f"Molecular stability: {validity_dict['mol_stable']:.3f}")
print(f"Validity: {molecular_metrics[0]:.3f}")
```

## Example Workflows

### Workflow 1: Converting Existing Data
```python
# 1. Convert your molecular data to ASE database format
from ase import Atoms
from ase.db import connect

db = connect('my_dataset.db')
for mol_data in your_molecular_data:
    mol = Atoms(symbols=mol_data['symbols'], positions=mol_data['positions'])
    db.write(mol, **mol_data['properties'])

# 2. Train the model
# python main_ase.py --ase_db_path my_dataset.db --dataset ase_dataset
```

### Workflow 2: Testing with QM9
```python
# 1. Convert QM9 to ASE format (if you have QM9 data)
from qm9.ase_database import create_ase_database_from_qm9
create_ase_database_from_qm9('qm9_train.npz', 'qm9_ase.db', 'train')

# 2. Compare results
# Train with original QM9 and ASE-converted QM9 to verify equivalence
```

## Testing

Run the validation tests to ensure everything works:

```bash
# Test basic ASE functionality
python example_ase_usage.py

# Run comprehensive validation
python test_ase_validation.py

# Test QM9 conversion (if QM9 data available)
python test_ase_conversion.py
```

## Technical Details

### Bond Order Calculation
- For QM9 datasets: Uses existing `get_bond_order` function
- For GEOM datasets: Uses existing `geom_predictor` function  
- For other datasets: Uses generic distance-based bonding rules

### Molecular Processing
- **RDKit**: Used when available (original functionality)
- **OpenBabel**: Used as fallback or when explicitly requested
- Both provide molecular validity, uniqueness, and novelty metrics

### Data Loading
- ASE databases are automatically split into train/validation/test sets
- Supports filtering by number of atoms
- Maintains compatibility with existing preprocessing pipeline

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install ase openbabel-wheel
   ```

2. **Database Path Issues**
   - Ensure the ASE database file exists
   - Use absolute paths when possible
   - Check file permissions

3. **Memory Issues**
   - Reduce batch size for large molecules
   - Filter by maximum number of atoms if needed

4. **Bond Order Calculation Failures**
   - The system falls back to generic distance-based bonding
   - Check atom symbols are recognized
   - Verify molecular geometries are reasonable

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check dataset statistics:
```python
from qm9.ase_database import ASEDatabaseReader
reader = ASEDatabaseReader('your_database.db')
dataset_info = reader.get_dataset_info('debug', with_h=True)
print(dataset_info)
```

## Performance Considerations

- ASE database loading is typically fast
- OpenBabel may be slower than RDKit for some operations
- Large databases should be split appropriately
- Consider using `num_workers > 0` for faster data loading

## Limitations

- OpenBabel SMILES generation may differ slightly from RDKit
- Some advanced RDKit features are not available in OpenBabel
- Bond order prediction is approximate for non-QM9/GEOM datasets
- 3D coordinate generation from SMILES may vary between tools

## Future Enhancements

Potential areas for improvement:
- Better bond order prediction for general molecules
- More sophisticated molecular descriptors
- Integration with other molecular databases
- Parallel processing for large datasets
- Enhanced error handling and validation