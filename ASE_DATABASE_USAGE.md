# ASE Database Support for E3 Diffusion

This extension adds support for general molecular datasets created using ASE's (Atomic Simulation Environment) database format to work with the E3 Diffusion framework, providing the same functionality as `main_qm9.py`.

## Features

- Load molecular datasets from ASE database files (`.db` format)
- Automatically detect atom types and generate dataset configurations
- Support for custom molecular properties from the database
- Train/validation/test data splitting
- Compatible with all existing E3 Diffusion models and training procedures
- Support for hydrogen removal option
- Property conditioning (using properties stored in the ASE database)

## Requirements

The ASE package is now included in `requirements.txt` and will be installed automatically:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To train E3 Diffusion on an ASE database:

```bash
python main_ase.py --datadir /path/to/your/database.db --exp_name my_ase_experiment
```

### Advanced Options

```bash
python main_ase.py \
    --datadir /path/to/your/database.db \
    --exp_name my_experiment \
    --max_molecules 10000 \
    --include_properties total_energy homo lumo gap \
    --conditioning total_energy homo \
    --remove_h \
    --batch_size 64 \
    --n_epochs 100
```

### Available Arguments

- `--datadir`: Path to ASE database file (required)
- `--max_molecules`: Maximum number of molecules to load (default: all)
- `--include_properties`: List of properties to include from database
- `--conditioning`: Properties to use for conditional generation
- `--remove_h`: Remove hydrogen atoms from the dataset
- `--filter_n_atoms`: Only use molecules with specific number of atoms

All other arguments are the same as `main_qm9.py`.

## Creating ASE Databases

You can create ASE databases using the ASE library:

```python
from ase import Atoms
from ase.db import connect

# Create database
db = connect('my_molecules.db')

# Add molecules with properties
atoms = Atoms('H2O', positions=[[0,0,0], [1,0,0], [0,1,0]])
data = {
    'total_energy': -76.4,
    'homo': -12.6,
    'lumo': 0.9,
    'gap': 13.5
}
db.write(atoms, data=data)
```

## Dataset Configuration

The system automatically generates dataset configurations based on the molecules in your ASE database:

- Detects all unique atom types
- Calculates maximum molecule size
- Generates statistics for visualization
- Creates appropriate atom encoders/decoders

## Property Conditioning

You can condition the generation on properties stored in your ASE database:

```bash
python main_ase.py \
    --datadir molecules.db \
    --conditioning total_energy homo lumo \
    --exp_name conditional_generation
```

This works the same way as QM9 property conditioning but uses your custom properties.

## Examples

### Water Molecules Dataset

```python
# Create a dataset of water molecules with varying properties
from ase import Atoms
from ase.db import connect
import numpy as np

db = connect('water_dataset.db')

for i in range(1000):
    # Slightly vary the water geometry
    positions = np.array([[0,0,0], [1,0,0], [0,1,0]]) + np.random.normal(0, 0.1, (3,3))
    atoms = Atoms('H2O', positions=positions)
    
    # Add properties
    data = {
        'binding_energy': np.random.uniform(-10, -8),
        'dipole_moment': np.random.uniform(1.5, 2.5)
    }
    db.write(atoms, data=data)
```

Then train:
```bash
python main_ase.py --datadir water_dataset.db --conditioning binding_energy
```

### Organic Molecules Dataset

```python
# Create a mixed organic molecule dataset
molecules = [
    ('CH4', methane_positions),      # Methane
    ('C2H6', ethane_positions),      # Ethane  
    ('C6H6', benzene_positions),     # Benzene
    ('NH3', ammonia_positions),      # Ammonia
]

db = connect('organic_molecules.db')

for symbols, positions in molecules:
    atoms = Atoms(symbols, positions=positions)
    data = {
        'formation_energy': calculate_formation_energy(atoms),
        'homo': calculate_homo(atoms),
        'lumo': calculate_lumo(atoms)
    }
    db.write(atoms, data=data)
```

## File Structure

The ASE support adds the following files:

- `main_ase.py`: Main training script for ASE datasets
- `qm9/data/prepare/ase_db.py`: ASE database processing functions
- `test_ase_support.py`: Test script for ASE functionality
- `example_ase_usage.py`: Example usage demonstration

## Limitations

- Properties must be numeric (int/float) values
- Very large molecules (>100 atoms) may require memory considerations
- Complex molecular properties may need preprocessing

## Troubleshooting

### Database Loading Issues
- Ensure your ASE database file exists and is readable
- Check that molecules in the database have valid atomic structures
- Verify property names don't conflict with ASE reserved keywords

### Memory Issues with Large Datasets
- Use `--max_molecules` to limit dataset size
- Consider splitting very large databases into smaller chunks
- Use `--batch_size` to control memory usage during training

### Property Conditioning Issues
- Ensure property names in `--conditioning` exist in your database
- Check that properties are numeric values
- Verify sufficient variation in property values for meaningful conditioning