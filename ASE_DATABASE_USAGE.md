# ASE Database Support for E3 Diffusion

This repository now supports loading molecular datasets from ASE (Atomic Simulation Environment) database format, in addition to the original QM9 and GEOM datasets.

## Installation

Make sure you have ASE installed:
```bash
pip install ase
```

This is already included in the updated `requirements.txt`.

## Usage

### Basic Usage

To train a model using an ASE database, use the `--dataset ase_db` flag along with the path to your database:

```bash
python main_qm9.py --dataset ase_db --ase_db_path /path/to/your/database.db
```

### Command Line Arguments

New arguments for ASE database support:

- `--dataset ase_db`: Use ASE database format
- `--ase_db_path`: Path to the ASE database file (required for ase_db dataset)
- `--split_ratios`: Train, validation, test split ratios as three numbers (default: 0.8 0.1 0.1)
- `--ase_to_eV`: JSON string for unit conversion factors (e.g., `'{"energy": 27.2114}'`)

### Examples

1. **Basic usage:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db
```

2. **Custom split ratios:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --split_ratios 0.7 0.2 0.1
```

3. **Remove hydrogen atoms:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --remove_h
```

4. **With unit conversion:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --ase_to_eV '{"energy": 27.2114, "homo": 27.2114, "lumo": 27.2114}'
```

### Creating an ASE Database

You can create an ASE database from molecular structures:

```python
from ase import Atoms
from ase.db import connect

# Create database
db = connect('molecules.db')

# Add molecules with properties
water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
db.write(water, data={'energy': -76.4, 'homo': -12.6, 'lumo': 1.4})

methane = Atoms('CH4', positions=[
    [0, 0, 0],        # C
    [1.089, 1.089, 1.089],   # H
    [1.089, -1.089, -1.089], # H
    [-1.089, 1.089, -1.089], # H
    [-1.089, -1.089, 1.089]  # H
])
db.write(methane, data={'energy': -40.5, 'homo': -14.4, 'lumo': 6.0})
```

### Supported Properties

The ASE loader automatically extracts molecular properties from the database. Common properties include:

- `energy`: Total energy
- `homo`: HOMO energy
- `lumo`: LUMO energy
- `gap`: HOMO-LUMO gap
- `mu`: Dipole moment
- `alpha`: Polarizability
- `zpve`: Zero-point vibrational energy
- `U0`, `U`, `H`, `G`, `Cv`: Thermodynamic properties

Any properties stored in the ASE database's `data` field will be automatically included.

### Features

- **Automatic atom type detection**: The loader automatically determines all atom types present in your dataset
- **Dynamic configuration**: Dataset parameters are updated based on actual data
- **Flexible splits**: Configurable train/validation/test ratios
- **Unit conversion**: Support for converting units (e.g., Hartree to eV)
- **Hydrogen handling**: Option to include or exclude hydrogen atoms
- **Full compatibility**: Works with all existing E3 diffusion model features

### Data Format

The ASE loader converts molecular structures to the same tensor format used by QM9:

- `positions`: Atomic coordinates (Å)
- `charges`: Atomic numbers
- `num_atoms`: Number of atoms per molecule
- Properties are stored as 1D tensors

The data is automatically padded and collated for efficient batch processing.