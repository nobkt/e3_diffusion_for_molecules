# Crystal Generation Usage Guide

This guide provides practical examples for using the molecular crystal generation system.

## Quick Start

### 1. Training a Crystal Generation Model

```bash
python main_crystal.py \
    --ase_db_path /path/to/crystals.db \
    --hidden_nf 128 \
    --n_layers 4 \
    --batch_size 4 \
    --epochs 100 \
    --exp_name my_crystal_model
```

**Arguments:**
- `--ase_db_path`: Path to ASE database containing crystal structures
- `--hidden_nf`: Hidden feature dimension (default: 128)
- `--n_layers`: Number of EGNN layers (default: 4)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--exp_name`: Name for this experiment (default: crystal_exp)

### 2. Generating Crystal Structures

After training, generate new crystal structures:

```bash
python sample_crystal.py \
    --model_path outputs/my_crystal_model/final_model.pt \
    --n_samples 10 \
    --n_atoms 20 \
    --output_dir generated_crystals \
    --output_format cif
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--n_samples`: Number of structures to generate (default: 10)
- `--n_atoms`: Number of atoms per structure (default: 20)
- `--output_dir`: Directory to save generated structures (default: generated_crystals)
- `--output_format`: Output format - 'cif', 'xyz', or 'both' (default: cif)

### 3. Evaluating Generated Structures

Evaluate the quality of generated structures:

```bash
python eval_crystal.py \
    --generated_dir generated_crystals \
    --reference_db /path/to/crystals.db \
    --format cif
```

**Arguments:**
- `--generated_dir`: Directory containing generated structures
- `--reference_db`: (Optional) Reference ASE database for comparison
- `--format`: Format of generated files - 'cif' or 'xyz' (default: cif)

## Data Preparation

### Creating an ASE Database from CIF Files

```python
from ase.io import read
from ase.db import connect

# Create database
db = connect('crystals.db', append=False)

# Add structures from CIF files
import glob
for cif_file in glob.glob('*.cif'):
    atoms = read(cif_file)
    db.write(atoms)

print(f"Added {len(db)} structures to database")
```

### Creating Test Database

```python
from ase import Atoms
from ase.db import connect
import numpy as np

db = connect('test_crystals.db', append=False)

# Create simple cubic crystals
for i in range(10):
    # Random lattice parameter
    a = np.random.uniform(5.0, 10.0)
    
    # Create simple cubic structure
    positions = np.random.rand(20, 3) * a
    atoms = Atoms(
        'H20',  # 20 hydrogen atoms
        positions=positions,
        cell=[a, a, a],
        pbc=True
    )
    db.write(atoms)

print(f"Created test database with {len(db)} structures")
```

## Testing the Implementation

### Test Periodic Utilities

```bash
python test_periodic_utils.py
```

Expected output:
```
Running periodic utilities tests...
✓ Coordinate conversion test passed
✓ Cell parameter conversion test passed
✓ Minimum image distance test passed
✓ Wrap to unit cell test passed
✓ Build neighbor list test passed

All tests passed! ✓
```

### Test Crystal Models

```bash
python test_crystal_models.py
```

Expected output:
```
Running crystal model tests...

Testing Periodic EGNN...
✓ Periodic EGNN test passed
Testing Lattice Diffusion...
✓ Lattice Diffusion test passed
Testing Crystal Dynamics...
✓ Crystal Dynamics test passed
Testing crystal sampling...
✓ Crystal sampling test passed

✅ All crystal model tests passed!
```

## Advanced Usage

### Using Fractional Coordinates

The system internally uses fractional coordinates for better handling of periodic boundaries:

```python
from crystal.data.periodic_utils import (
    cartesian_to_fractional,
    fractional_to_cartesian
)

# Convert Cartesian to fractional
fractional = cartesian_to_fractional(positions, cell_vectors)

# Convert back to Cartesian
cartesian = fractional_to_cartesian(fractional, cell_vectors)
```

### Computing Periodic Distances

```python
from crystal.data.periodic_utils import minimum_image_distance

# Compute distances with periodic boundary conditions
distances, vectors = minimum_image_distance(
    positions1, positions2, cell_vectors,
    use_fractional=True
)
```

### Loading Crystal Structures

```python
from crystal.data.crystal_loader import CrystalDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = CrystalDataset(
    db_path='crystals.db',
    use_fractional_coords=True,
    max_atoms=100
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_crystal_batch
)
```

## Model Architecture

The crystal generation system consists of three main components:

1. **Periodic EGNN**: E(3) equivariant graph neural network with periodic boundary conditions
2. **Lattice Diffusion**: Diffusion model for lattice parameters (a, b, c, α, β, γ)
3. **Crystal Dynamics**: Integrated model combining position and lattice dynamics

### Key Features

- ✅ Handles periodic boundary conditions correctly
- ✅ Uses fractional coordinates for lattice-invariant representation
- ✅ Learns both atomic positions and lattice parameters
- ✅ E(3) equivariant message passing
- ✅ Physical constraints on lattice parameters

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or number of atoms:
```bash
python main_crystal.py --batch_size 2 --max_atoms 50 ...
```

### Database Not Found

Ensure the ASE database file exists:
```bash
python -c "from ase.db import connect; db = connect('crystals.db'); print(f'Found {len(db)} structures')"
```

### Model Not Loading

Check that the model architecture matches:
```bash
python sample_crystal.py \
    --model_path path/to/model.pt \
    --hidden_nf 128 \  # Must match training
    --n_layers 4 \     # Must match training
    ...
```

## References

For more details on the implementation, see:
- `ARCHITECTURE_DIAGRAM.md` - System architecture
- `MOLECULAR_CRYSTAL_DESIGN.md` - Detailed design
- `MOLECULAR_CRYSTAL_SPECIFICATION.md` - Requirements specification

## Citation

If you use this code, please cite:

```bibtex
@software{crystal_generation,
  title={Molecular Crystal Generation with E(3) Equivariant Diffusion},
  author={Your Name},
  year={2025}
}
```
