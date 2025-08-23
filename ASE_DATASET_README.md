# ASE Database Dataset Loading

This document describes the ASE (Atomic Simulation Environment) database loading functionality added to the e3_diffusion_for_molecules repository.

## Overview

The ASE dataset loading feature allows you to load molecular datasets directly from ASE database files (`.db` format) for use with the diffusion model pipeline. This provides a convenient way to work with molecular data stored in ASE's standardized database format.

## Features

- Load molecular structures from ASE database files
- Automatic train/validation/test splitting
- Support for filtering by molecule size
- Compatible with existing diffusion model pipeline
- Supports both hydrogen-inclusive and hydrogen-exclusive configurations
- Configurable batch processing (sequential or random)

## Installation

The ASE package is now included in the requirements. Install it with:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Configuration

To use ASE dataset loading, configure your dataset settings as follows:

```python
from types import SimpleNamespace
from qm9.dataset import retrieve_dataloaders

cfg = SimpleNamespace(
    dataset='ase',                              # Use ASE dataset
    ase_db_file='/path/to/your/molecules.db',   # Path to ASE database file
    ase_max_entries=None,                       # Optional: limit number of molecules
    remove_h=False,                             # Whether to remove hydrogen atoms
    include_charges=False,                      # Whether to include atomic charges
    device=torch.device('cuda'),                # Device for computations
    sequential=False,                           # Use random batching
    batch_size=32,                              # Batch size
    filter_molecule_size=None                   # Optional: filter by atom count
)

# Load dataloaders
dataloaders, charge_scale = retrieve_dataloaders(cfg)
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | str | Must be set to `'ase'` to use ASE loading |
| `ase_db_file` | str | Path to the ASE database file (.db) |
| `ase_max_entries` | int/None | Maximum number of molecules to load (None = all) |
| `remove_h` | bool | Whether to exclude hydrogen atoms |
| `include_charges` | bool | Whether to include atomic charges in the data |
| `device` | torch.device | Device for tensor operations |
| `sequential` | bool | Sequential vs random batch processing |
| `batch_size` | int | Number of molecules per batch |
| `filter_molecule_size` | int/None | Keep only molecules with specific atom count |

### Creating ASE Database Files

You can create ASE database files using the ASE library:

```python
import ase.db
from ase import Atoms

# Connect to database
db = ase.db.connect('molecules.db')

# Add molecules
atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
db.write(atoms)

# Add more molecules...
```

### Example Usage

See `example_ase_usage.py` for a complete working example that demonstrates:
- Creating an example ASE database
- Loading the dataset with the diffusion model pipeline
- Processing batches of molecular data

```bash
python example_ase_usage.py
```

## Data Format

The ASE dataset loader expects database entries containing:
- Atomic positions (xyz coordinates)
- Atomic numbers (element types)

The loader automatically converts this to the format expected by the diffusion model:
- `positions`: Atomic coordinates `[batch, n_atoms, 3]`
- `one_hot`: Atom type encoding `[batch, n_atoms, n_types]`
- `atom_mask`: Valid atom mask `[batch, n_atoms]`
- `edge_mask`: Valid edge mask `[batch*n_atoms*n_atoms, 1]`
- `charges`: Atomic charges (if enabled) `[batch, n_atoms, 1]`

## Supported Atom Types

The default configuration supports common organic chemistry atoms:

**With hydrogens:**
- H (Hydrogen), C (Carbon), N (Nitrogen), O (Oxygen), F (Fluorine), P (Phosphorus), S (Sulfur), Cl (Chlorine)

**Without hydrogens:**
- C (Carbon), N (Nitrogen), O (Oxygen), F (Fluorine), P (Phosphorus), S (Sulfur), Cl (Chlorine)

To support additional atom types, modify the configurations in `configs/datasets_config.py`.

## Data Splitting

The loader automatically splits data into train/validation/test sets:
- Training: 80% of data (default)
- Validation: 10% of data
- Test: 10% of data

The splitting is deterministic (uses random seed 42) for reproducible results.

## Testing

Run the test suite to verify the ASE dataset functionality:

```bash
python test_ase_dataset.py
```

## Files Added/Modified

- `build_ase_dataset.py`: Core ASE dataset loading functionality
- `configs/datasets_config.py`: ASE dataset configurations
- `qm9/dataset.py`: Integration with main dataset loading pipeline
- `requirements.txt`: Added ASE dependency
- `test_ase_dataset.py`: Test suite for ASE functionality
- `example_ase_usage.py`: Example usage script

## Troubleshooting

**Error: "ASE database file not found"**
- Ensure the path to your .db file is correct
- Check that the file exists and is readable

**Error: "No data available after filtering"**
- Your filter_molecule_size setting may be too restrictive
- Check that your database contains molecules with the specified atom count

**Error: "Bad key" when creating ASE database**
- ASE databases have restrictions on key names
- Use simple key names or no additional properties when writing to the database

**Memory issues with large databases**
- Use `ase_max_entries` to limit the number of molecules loaded
- Reduce `batch_size` for memory-constrained environments
- Consider using `sequential=True` for more memory-efficient processing

## Integration with Existing Models

The ASE dataset loader is fully compatible with the existing diffusion model pipeline. Simply replace your dataset configuration with ASE settings and the rest of your training/evaluation code should work unchanged.

The output format matches exactly what the model expects from QM9 and GEOM datasets, ensuring seamless integration.