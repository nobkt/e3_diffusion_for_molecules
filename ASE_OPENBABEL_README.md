# ASE Database and OpenBabel Integration

This extension adds support for general molecular datasets using ASE (Atomic Simulation Environment) databases and OpenBabel for molecular operations, replacing the previous dependence on RDKit and SMILES representations.

## Features

### 🔧 New Functionality

- **ASE Database Support**: Load molecular datasets from ASE database files
- **OpenBabel Integration**: Use OpenBabel for molecular operations instead of RDKit
- **General Molecule Support**: Handle molecules that cannot be represented as SMILES
- **Automatic Dataset Configuration**: Generate dataset configurations automatically from database content
- **Backwards Compatibility**: Existing QM9 and GEOM workflows remain unchanged

### 🧪 Key Components

1. **`qm9/openbabel_functions.py`**: OpenBabel-based molecular functions
2. **`qm9/ase_dataset.py`**: ASE database loading and dataset creation
3. **Enhanced analysis and visualization**: Updated to support general molecules

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install ase openbabel-wheel
```

### Basic Usage

```python
from qm9.dataset import retrieve_dataloaders

# Configure for ASE database
class Config:
    dataset = 'ase_general'  # Any name containing 'ase'
    ase_db_path = '/path/to/your/molecules.db'
    batch_size = 32
    num_workers = 0
    max_atoms = 100
    include_charges = True

config = Config()
dataloaders, charge_scale = retrieve_dataloaders(config)

# Use the dataloaders as normal
for batch in dataloaders['train']:
    positions = batch['positions']  # Shape: [batch_size, max_atoms, 3]
    atom_types = batch['one_hot']   # Shape: [batch_size, max_atoms, n_atom_types]
    masks = batch['atom_mask']      # Shape: [batch_size, max_atoms]
    # ... your training code
```

### Creating ASE Databases

```python
from ase import Atoms
from ase.db import connect

# Create database
db = connect('molecules.db')

# Add molecules
water = Atoms('OH2', positions=[[0,0,0], [1,0,0], [0,1,0]])
db.write(water)

methane = Atoms('CH4', positions=[[0,0,0], [1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0]])
db.write(methane)

print(f"Database contains {len(db)} molecules")
```

### Molecular Analysis with OpenBabel

```python
from qm9.openbabel_functions import BasicMolecularMetricsOB, build_molecule_ob
from qm9.analyze import analyze_stability_for_molecules

# Create dataset info (automatically generated for ASE datasets)
dataset_info = get_ase_dataset_info('molecules.db')

# Analyze generated molecules
metrics = BasicMolecularMetricsOB(dataset_info)
generated_molecules = [(positions, atom_types), ...]  # Your generated molecules

# Get validity, uniqueness, and novelty
results = metrics.evaluate(generated_molecules)
validity, uniqueness, novelty = results[0]

print(f"Validity: {validity:.2%}")
print(f"Uniqueness: {uniqueness:.2%}")
print(f"Novelty: {novelty:.2%}")
```

## Detailed Examples

### Complete Training Example

```python
import torch
from qm9.ase_dataset import create_sample_ase_db, load_ase_dataset

# 1. Create or use existing ASE database
db_path = 'training_molecules.db'
create_sample_ase_db(db_path, num_molecules=1000)

# 2. Configure dataset loading
class TrainingConfig:
    dataset = 'ase_training'
    ase_db_path = db_path
    batch_size = 64
    num_workers = 4
    max_atoms = 50
    include_charges = True
    
    # Training parameters
    n_epochs = 100
    lr = 1e-4
    # ... other training parameters

config = TrainingConfig()

# 3. Load dataset
dataloaders, _ = load_ase_dataset(db_path, config)

# 4. Training loop (simplified)
model = YourDiffusionModel(dataset_info=dataloaders['train'].dataset_info)

for epoch in range(config.n_epochs):
    for batch in dataloaders['train']:
        # Your training code here
        loss = model(batch)
        loss.backward()
        # ...
```

### Custom Molecule Analysis

```python
from qm9.openbabel_functions import build_molecule_ob, mol_to_xyz_string

# Build molecule from coordinates and atom types
positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])  # Two atoms
atom_types = torch.tensor([1, 1])  # Two carbons (if C is index 1)

mol = build_molecule_ob(positions, atom_types, dataset_info)

if mol is not None:
    print(f"Molecule has {mol.NumAtoms()} atoms and {mol.NumBonds()} bonds")
    
    # Convert to XYZ format
    xyz_string = mol_to_xyz_string(mol)
    print("XYZ format:")
    print(xyz_string)
    
    # Get molecular formula
    formula = mol.GetFormula()
    print(f"Formula: {formula}")
```

## API Reference

### Key Classes and Functions

#### `BasicMolecularMetricsOB`
OpenBabel-based molecular metrics calculator.

```python
metrics = BasicMolecularMetricsOB(dataset_info, dataset_fingerprints=None)
validity, uniqueness, novelty = metrics.evaluate(generated_molecules)
```

#### `ASEDataset` and `ASEDataLoader`
Dataset classes for ASE database files.

```python
dataset = ASEDataset(db_path, dataset_info, max_atoms=100, include_charges=True)
loader = ASEDataLoader(dataset, batch_size=32, shuffle=True)
```

#### `get_ase_dataset_info(db_path, max_atoms=100, name="ase_general")`
Automatically generate dataset configuration from ASE database.

#### `create_sample_ase_db(db_path, num_molecules=100)`
Create a sample ASE database for testing.

### Dataset Configuration

ASE datasets are automatically configured with:
- `atom_encoder`: Mapping from element symbols to indices
- `atom_decoder`: List of element symbols
- `max_n_nodes`: Maximum number of atoms
- `n_nodes`: Distribution of molecule sizes
- `colors_dic` and `radius_dic`: For visualization
- `is_ase: True`: Flag indicating ASE dataset

## Backwards Compatibility

The new functionality is fully backwards compatible:

- QM9 and GEOM datasets work exactly as before
- RDKit-based functions are still available when RDKit is installed
- Existing scripts and configurations require no changes
- OpenBabel functions are used as fallbacks when RDKit is not available

## Error Handling

The implementation includes robust error handling:

- Missing elements are handled with default bond configurations
- Invalid molecules are skipped during loading
- OpenBabel warnings are suppressed for common valence issues
- Graceful fallbacks for unsupported operations

## Performance Considerations

- **Dataset Loading**: ASE database loading is optimized for large datasets
- **Memory Usage**: Molecules are loaded on-demand to conserve memory
- **Bond Prediction**: General bond order prediction may be slower than QM9-specific functions
- **Molecular Metrics**: OpenBabel-based metrics use fingerprints instead of SMILES for uniqueness

## Troubleshooting

### Common Issues

1. **"No molecules loaded"**: Check ASE database format and element symbols
2. **"Unknown element"**: Add custom element to bond prediction functions
3. **Memory issues**: Reduce `max_atoms` or `batch_size` parameters
4. **Slow loading**: Increase `num_workers` for parallel processing

### Debug Mode

Enable debugging for dataset loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed loading information
dataloaders, _ = load_ase_dataset(db_path, config)
```

## Contributing

To extend the functionality:

1. **Add new elements**: Update `predict_bond_order_general()` and `allowed_bonds`
2. **Custom metrics**: Inherit from `BasicMolecularMetricsOB`
3. **New datasets**: Follow the ASE dataset pattern
4. **Visualization**: Update `visualizer.py` for custom rendering

## Citation

If you use this extension in your research, please cite both the original E3 diffusion paper and acknowledge the ASE and OpenBabel libraries.

## License

This extension follows the same license as the original repository.